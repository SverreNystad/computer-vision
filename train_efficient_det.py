import os
import getpass
import optuna
import wandb
import torch
import torch.optim as optim
from effdet import get_efficientdet_config, EfficientDet
from effdet.efficientdet import DetBenchPredict  # inference wrapper
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from optuna.integration.wandb import WeightsAndBiasesCallback
from codecarbon import track_emissions, OfflineEmissionsTracker
from dotenv import load_dotenv
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import sys, logging
from optuna.study import StudyDirection

load_dotenv()

# Setup environment and wandb directories
USER_NAME = getpass.getuser()
DEVICE_PATH = os.getenv("DEVICE_PATH") or "/work"
os.environ["WANDB_DIR"] = f"{DEVICE_PATH}/{USER_NAME}/"
os.environ["WANDB_CACHE_DIR"] = f"{DEVICE_PATH}/{USER_NAME}/.cache/"
os.environ["WANDB_CONFIG_DIR"] = f"{DEVICE_PATH}/{USER_NAME}/.config/wandb"
os.environ["WANDB_DATA_DIR"] = f"{DEVICE_PATH}/{USER_NAME}/.cache/wandb-data/"
os.environ["WANDB_ARTIFACT_DIR"] = f"{DEVICE_PATH}/{USER_NAME}/artifacts"

if not os.path.exists(os.getenv("WANDB_CONFIG_DIR")):
    os.makedirs(os.getenv("WANDB_CONFIG_DIR"))

WANDB_API_KEY = os.getenv("WANDB_API_KEY")
print(f"WANDB_API_KEY: {WANDB_API_KEY}")
wandb.login(key=WANDB_API_KEY)


class OurDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.img_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith((".jpg", ".png"))])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(self.label_dir, label_file)

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        boxes = []
        labels = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                labels.append(int(parts[0]))
                boxes.append(list(map(float, parts[1:])))
        targets = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 4)),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.empty((0,), dtype=torch.int64)
        }
        return img, targets


def convert_box_format(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes
    x_center, y_center, w, h = boxes.unbind(-1)
    xmin = x_center - w / 2
    ymin = y_center - h / 2
    xmax = x_center + w / 2
    ymax = y_center + h / 2
    return torch.stack([xmin, ymin, xmax, ymax], dim=-1)


def objective(trial: optuna.Trial):
    run = wandb.init(project="cv-efficientdet", reinit=True)

    num_epochs = trial.suggest_int("epochs", 1, 6000)
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.001)

    transform = transforms.Compose([transforms.ToTensor()])
    train_img_dir = f"{DEVICE_PATH}/{USER_NAME}/computer-vision/data/rgb/train/images"
    train_label_dir = f"{DEVICE_PATH}/{USER_NAME}/computer-vision/data/rgb/train/labels"
    val_img_dir = f"{DEVICE_PATH}/{USER_NAME}/computer-vision/data/rgb/val/images"
    val_label_dir = f"{DEVICE_PATH}/{USER_NAME}/computer-vision/data/rgb/val/labels"

    train_dataset = OurDataset(train_img_dir, train_label_dir, transform=transform)
    val_dataset = OurDataset(val_img_dir, val_label_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

    model_name = 'tf_efficientdet_d2'
    config = get_efficientdet_config(model_name)
    config.num_classes = 1  
    model = EfficientDet(config, pretrained_backbone=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_loss = float('inf')

    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "lr": learning_rate,
            "weight_decay": weight_decay
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        print(f"Epoch {epoch + 1}/{num_epochs}: Train {avg_train_loss:.4f}, Val {avg_val_loss:.4f}")

    inference_model = DetBenchPredict(model)
    inference_model.eval()

    map50_metric = MeanAveragePrecision(iou_thresholds=[0.5])
    map95_metric = MeanAveragePrecision(iou_thresholds=[0.95])
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            preds = inference_model(images)
            for pred, target in zip(preds, targets):
                gt_boxes = target["boxes"].to(device)
                gt_boxes = convert_box_format(gt_boxes) if gt_boxes.numel() > 0 else gt_boxes
                all_targets.append({
                    "boxes": gt_boxes,
                    "labels": target["labels"].to(torch.int64)
                })
                pred_boxes = pred["boxes"].to(device) if "boxes" in pred else torch.empty((0, 4), device=device)
                pred_boxes = convert_box_format(pred_boxes) if pred_boxes.numel() > 0 else pred_boxes
                all_preds.append({
                    "boxes": pred_boxes,
                    "scores": pred["scores"].to(device) if "scores" in pred else torch.empty((0,), device=device),
                    "labels": pred["labels"].to(torch.int64) if "labels" in pred else torch.empty((0,), device=device)
                })

    map50_metric.update(all_preds, all_targets)
    map95_metric.update(all_preds, all_targets)
    results_map50 = map50_metric.compute()
    results_map95 = map95_metric.compute()

    wandb.log({
        "mAP_50": results_map50["map"] if "map" in results_map50 else results_map50,
        "mAP_95": results_map95["map"] if "map" in results_map95 else results_map95,
    })
    run.finish()

    return results_map95["map"] if "map" in results_map95 else results_map95


#@track_emissions(offline=True, country_iso_code="NOR")
def main(study_name: str):
    tracker = OfflineEmissionsTracker(country_iso_code="NOR")
    try:
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        MYSQL_USER = os.getenv("MYSQL_USER")
        MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
        storage_name = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@mysql.stud.ntnu.no/timma_tdt4265_db"
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            load_if_exists=True,
            directions=[StudyDirection.MAXIMIZE],
        )
        wandb_kwargs = {"project": study_name}
        wandb_callback = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs, as_multirun=True)
        for _ in range(100):
            study.optimize(objective, n_trials=10, callbacks=[wandb_callback])
            tracker.flush()
        print("Best trial:", study.best_trial)
    finally:
        tracker.stop()


if __name__ == "__main__":
    study_name = "cv-efficientdet"
    main(study_name)

