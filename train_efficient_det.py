import os
import getpass
import optuna
import wandb
import torch
import torch.optim as optim
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchPredict
from effdet import DetBenchPredict  # inference wrapper
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import v2
from PIL import Image
from tqdm import tqdm
from optuna.integration.wandb import WeightsAndBiasesCallback
from codecarbon import track_emissions, OfflineEmissionsTracker
from dotenv import load_dotenv
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import sys, logging
from optuna.study import StudyDirection
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

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

#def collate_fn(batch):
#    images, targets = zip(*batch)
#    images = torch.stack(images, dim=0)
#    return images, list(targets) 

class OurDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform):
        self.img_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_files = sorted([f for f in os.listdir(self.img_dir) if f.lower().endswith((".jpg", ".png"))])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(self.label_dir, label_file)

        img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32)

        boxes = []
        labels = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                labels.append(int(parts[0]))
                boxes.append(list(map(float, parts[1:])))

        transformed = self.transform(image=img, bboxes=boxes, labels=labels)
        img = transformed["image"]
        boxes = transformed["bboxes"]
        labels = transformed["labels"]

        targets = {
            "bbox": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 4)),
            "cls": torch.tensor(labels, dtype=torch.int64) if labels else torch.empty((0,), dtype=torch.int64),
            "img_scale": 1.0,
            "img_size": torch.tensor([img.size(dim=1), img.size(dim=2)], dtype=torch.float32),
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


def objective(trail: optuna.Trial):
    num_epochs = 1# trail.suggest_int("epochs", 1, 2)
    learning_rate = trail.suggest_float("lr", 1e-5, 1e-1, log=True)
    weight_decay = trail.suggest_float("weight_decay", 0.0, 0.001)
    brightness = trail.suggest_float("brightness", 0.0, 1.0)
    hue = trail.suggest_float("hue", 0.0, 0.5)
    saturation = trail.suggest_float("saturation", 0.0, 1.0)
    contrast = trail.suggest_float("contrast", 0.0, 1.0)
    rotation = trail.suggest_float("rotation", 0.0, 45.0)
    translate_x = trail.suggest_float("translate_x", 0.0, 1.0)
    translate_y = trail.suggest_float("translate_y", 0.0, 1.0)
    scale = trail.suggest_float("scale", 0.0, 0.9)
    shear = trail.suggest_float("shear", 0.0, 10.0)
    flipud = trail.suggest_float("flipud", 0.2, 0.8)

    transform = A.Compose([
        A.Resize(768, 768),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=flipud),
        A.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue),
        A.Affine(rotate=rotation, translate_percent={'x': translate_x, 'y': translate_y},
                 scale=(1.0 - scale, 1.0 + scale), shear=shear),
        A.Perspective(p=0.00012),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['labels']))

    train_img_dir = f"{DEVICE_PATH}/{USER_NAME}/computer-vision/data/rgb/images/train"
    train_label_dir = f"{DEVICE_PATH}/{USER_NAME}/computer-vision/data/rgb/labels/train"
    val_img_dir = f"{DEVICE_PATH}/{USER_NAME}/computer-vision/data/rgb/images/valid"
    val_label_dir = f"{DEVICE_PATH}/{USER_NAME}/computer-vision/data/rgb/labels/valid"

    train_dataset = OurDataset(train_img_dir, train_label_dir, transform=transform)
    val_dataset = OurDataset(val_img_dir, val_label_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    model_name = 'tf_efficientdet_d2'
    config = get_efficientdet_config(model_name)
    config.num_classes = 1  
    base_model = EfficientDet(config, pretrained_backbone=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    base_model.to(device)

    train_model = DetBenchTrain(base_model)
    train_model.to(device)

    optimizer = optim.AdamW(train_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_loss = float('inf')

    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        train_model.train()
        running_loss = 0.0
        for images, targets in tqdm(iter(train_loader), desc="Training on images"):
            images = images.to(device)#torch.stack(images, dim=0)#images.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}

            optimizer.zero_grad()
            loss_dict = train_model(images, targets)
            loss = loss_dict["loss"]#sum(loss for loss in loss_dict.values())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)

        train_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in tqdm(iter(val_loader), desc="Validating on images"):
                images = images.to(device)#torch.stack(images, dim=0)#images.to(device)
                targets = {k: v.to(device) for k, v in targets.items()}

                loss_dict = train_model(images, targets)
                loss = loss_dict["loss"]#sum(loss for loss in loss_dict.values())
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

    inference_model = DetBenchPredict(base_model)
    inference_model.to(device)
    inference_model.eval()

    map50_metric = MeanAveragePrecision(iou_thresholds=[0.5])
    map95_metric = MeanAveragePrecision(iou_thresholds=[0.95])
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, target in tqdm(iter(val_loader), desc="Final validation"):
            images = images.to(device)#torch.stack(images, dim=0)
            target = {k: v.to(device) for k, v in target.items()}
            pred = inference_model(images)
            pred = pred.squeeze(0)
            pred.to(device)
            if pred.numel() == 0:
                boxes = torch.empty((0, 4), device=device)
                boxes.to(device)
                scores = torch.empty((0,), device=device)
                scores.to(device)
                labels = torch.empty((0,), dtype=torch.int64, device=device)
                labels.to(device)
            else:
                # Assuming pred is of shape [N, 6]: [x1, y1, x2, y2, score, label]
                boxes = pred[:, :4]
                boxes.to(device)
                scores = pred[:, 4]
                scores.to(device)
                labels = pred[:, 5].long()  # class labels as integers
                labels.to(device)

            all_preds.append({
                "boxes": boxes,
                "scores": scores,
                "labels": labels
            })

            gt_boxes = target["bbox"]
            if gt_boxes.numel() > 0:
                gt_boxes = convert_box_format(gt_boxes)
            gt_labels = target["cls"]
            if gt_labels.ndim > 1:
                gt_labels = gt_labels.view(-1)

            all_targets.append({
                "boxes": gt_boxes.to(device),
                "labels": gt_labels.to(device)
            })

    map50_metric.update(all_preds, all_targets)
    map95_metric.update(all_preds, all_targets)
    results_map50 = map50_metric.compute()
    results_map95 = map95_metric.compute()

    wandb.log({
        "mAP_50": results_map50["map"] if "map" in results_map50 else results_map50,
        "mAP_95": results_map95["map"] if "map" in results_map95 else results_map95,
    })

    return results_map95["map"] if "map" in results_map95 else results_map95


#@track_emissions(offline=True, country_iso_code="NOR")
def main(study_name: str):
    tracker = OfflineEmissionsTracker(country_iso_code="NOR")
    run = wandb.init(entity="sverrenystad-ntnu", project=study_name, reinit=True)
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
        run.finish()


if __name__ == "__main__":
    study_name = "cv-efficientdet"
    main(study_name)

