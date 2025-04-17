import os
import getpass
import optuna
import wandb
import torch
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import v2
import torchvision.utils as tu
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
from svetim_detector import get_custom_detector

load_dotenv()
STUDY_NAME = "cv-svetimdet-v5"

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

def yolo_to_xyxy(box, img_width, img_height):
    """
    Convert a single bounding box from YOLO format to xyxy format.

    Args:
        box (list or tuple): [x_center, y_center, width, height] in normalized coordinates.
        img_width (int): Image width in pixels.
        img_height (int): Image height in pixels.

    Returns:
        list: Bounding box in [xmin, ymin, xmax, ymax] format in absolute pixel coordinates.
    """
    x_center, y_center, w, h = box
    xmin = (x_center - w / 2) * img_width
    ymin = (y_center - h / 2) * img_height
    xmax = (x_center + w / 2) * img_width
    ymax = (y_center + h / 2) * img_height
    return [xmin, ymin, xmax, ymax]

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

        img = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)

        boxes = []
        labels = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                labels.append(int(parts[0])+1)
                boxes.append(list(map(float, parts[1:])))

        transformed = self.transform(image=img, bboxes=boxes, labels=labels)
        img = transformed["image"]
        boxes = transformed["bboxes"]
        labels = transformed["labels"]

        img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0)#transforms.ToTensor()(img)

        _, img_height, img_width = img.shape
        boxes = [yolo_to_xyxy(b, img_width, img_height) for b in boxes]

        targets = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 4)),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.empty((0,), dtype=torch.int64),
        }
        return img, targets

def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

def convert_box_format(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes
    x_center, y_center, w, h = boxes.unbind(-1)
    xmin = x_center - w / 2
    ymin = y_center - h / 2
    xmax = x_center + w / 2
    ymax = y_center + h / 2
    return torch.stack([xmin, ymin, xmax, ymax], dim=-1)

wandb_kwargs = {"entity": "sverrenystad-ntnu", "project": STUDY_NAME}
wandb_callback = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs, as_multirun=True)

@wandb_callback.track_in_wandb()
def objective(trail: optuna.Trial):
    should_save_images = False 

    num_epochs = 5_000#trail.suggest_int("epochs", 1, 6000)
    learning_rate = trail.suggest_float("lr", 1e-5, 1e-1, log=True)
    weight_decay = trail.suggest_float("weight_decay", 0.0, 0.001)
    brightness = trail.suggest_float("brightness", 0.0, 1.0)
    hue = trail.suggest_float("hue", 0.0, 0.5)
    saturation = trail.suggest_float("saturation", 0.0, 1.0)
    contrast = trail.suggest_float("contrast", 0.0, 1.0)
    rotation = trail.suggest_float("rotation", 0.0, 45.0)
    translate_x = trail.suggest_float("translate_x", 0.0, 0.25)
    translate_y = trail.suggest_float("translate_y", 0.0, 0.25)
    scale = trail.suggest_float("scale", 0.0, 0.5)
    shear = trail.suggest_float("shear", 0.0, 10.0)
    flipud = trail.suggest_float("flipud", 0.2, 0.8)

    wandb.log({
        "parameters/num_epochs": num_epochs,
        "parameters/lr": learning_rate,
        "parameters/weight_decay": weight_decay,
        "parameters/brightness": brightness,
        "parameters/hue": hue,
        "parameters/saturation": saturation,
        "parameters/contrast": contrast,
        "parameters/rotation": rotation,
        "parameters/translate_x": translate_x,
        "parameters/translate_y": translate_y,
        "parameters/scale": scale,
        "parameters/shear": shear,
        "parameters/flipud": flipud,
    })

    transform = A.Compose([
        A.Resize(1024, 1024),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=flipud),
        A.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue),
        A.Affine(rotate=rotation, translate_percent={'x': translate_x, 'y': translate_y},
                 scale=(1.0 - scale, 1.0 + scale), shear=shear),
        A.Perspective(p=0.00012),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['labels']))

    train_img_dir = f"{DEVICE_PATH}/{USER_NAME}/computer-vision/data/rgb/images/train"
    train_label_dir = f"{DEVICE_PATH}/{USER_NAME}/computer-vision/data/rgb/labels/train"
    val_img_dir = f"{DEVICE_PATH}/{USER_NAME}/computer-vision/data/rgb/images/valid"
    val_label_dir = f"{DEVICE_PATH}/{USER_NAME}/computer-vision/data/rgb/labels/valid"

    train_dataset = OurDataset(train_img_dir, train_label_dir, transform=transform)
    val_dataset = OurDataset(val_img_dir, val_label_dir, transform=transform)
    n_rows = 2
    n_cols = 2
    train_loader = DataLoader(train_dataset, batch_size=n_rows*n_cols, shuffle=True, num_workers=2, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=n_rows*n_cols, shuffle=False, num_workers=2, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_custom_detector() 
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_loss = float('inf')

    final_map95 = 0.0
    final_map50 = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, targets in tqdm(train_loader, desc=f"Training on images in epoch {epoch+1}/{num_epochs}"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Average Training Loss = {avg_train_loss:.4f}")

        #model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Validating on images in epoch {epoch+1}/{num_epochs}"):
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()

        avg_val_loss = val_loss / len(val_loader)

        metric_map50 = MeanAveragePrecision(iou_thresholds=[0.5])
        metric_map95 = MeanAveragePrecision(iou_thresholds=[0.95])
        all_preds = []
        all_targets = []
        wandb_true_images = []
        wandb_pred_images = []


        model.eval()
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Evaluation Epoch {epoch+1}/{num_epochs}"):
                images = [img.to(device) for img in images]
                # In eval mode, the model returns predictions as a list of dictionaries.
                predictions = model(images)
                # For each image, create ground truth dictionaries expected by torchmetrics.
                for i, pred in enumerate(predictions):
                    gt = {
                        "boxes": targets[i]["boxes"].to(device),
                        "labels": targets[i]["labels"].to(device)
                    }
                    all_preds.append(pred)
                    all_targets.append(gt) 


                if should_save_images:
                    predicted_images = []
                    true_images = []
                    for i, img in enumerate(images):
                        predicted_images.append(tu.draw_bounding_boxes(img, predictions[i]["boxes"], colors="red").cpu())
                        true_images.append(tu.draw_bounding_boxes(img, targets[i]["boxes"].to(device), colors="red").cpu())

                    pred_imgs = tu.make_grid(predicted_images, nrow=n_rows)
                    true_imgs = tu.make_grid(true_images, nrow=n_rows)

                    pred_imgs = pred_imgs.cpu().numpy().transpose(1, 2, 0).astype(np.float32)
                    true_imgs = true_imgs.cpu().numpy().transpose(1, 2, 0).astype(np.float32)

                    wandb_true_images.append(wandb.Image(true_imgs, caption="True boxes and labels"))
                    wandb_pred_images.append(wandb.Image(pred_imgs, caption="Predicted boxes and labels"))

        metric_map50.update(all_preds, all_targets)
        metric_map95.update(all_preds, all_targets)
        results_map50 = metric_map50.compute()
        results_map95 = metric_map95.compute()

        if should_save_images:
            wandb.summary["predicted_images"] = wandb_pred_images
            wandb.summary["true_images"] = wandb_true_images

        # print(f"Epoch {epoch+1}: mAP@0.5 = {results_map50['map']:.4f}, mAP@0.95 = {results_map95['map']:.4f}")

        # Log epoch metrics to WANDB.
        wandb.log({
            "train/loss": avg_train_loss,
            "val/loss": avg_val_loss,
            "metrics/mAP50": results_map50["map"],
            "metrics/mAP95": results_map95["map"],
        })

        # Save final epoch’s mAP@0.95 to be returned as the objective value.
        final_map95 = results_map95["map"]
        final_map50 = results_map50["map"]

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        print(f"Epoch {epoch + 1}/{num_epochs}: Train {avg_train_loss:.4f}, Val {avg_val_loss:.4f}, mAP@0.5 = {results_map50['map']:.4f}, mAP@0.95 = {results_map95['map']:.4f}")


    return final_map95, final_map50


#@track_emissions(offline=True, country_iso_code="NOR")
def main(study_name: str):
    tracker = OfflineEmissionsTracker(country_iso_code="NOR")
    try:
        #wandb.init(entity="sverrenystad-ntnu", project=STUDY_NAME, reinit=True)
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        MYSQL_USER = os.getenv("MYSQL_USER")
        MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
        storage_name = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@mysql.stud.ntnu.no/timma_tdt4265_db"
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            load_if_exists=True,
            directions=[StudyDirection.MAXIMIZE, StudyDirection.MAXIMIZE],
        )
        for _ in range(100):
            study.optimize(objective, n_trials=10, callbacks=[wandb_callback])
            tracker.flush()
            wandb.finish()
        print("Best trial:", study.best_trial)
    finally:
        tracker.stop()


if __name__ == "__main__":
    main(STUDY_NAME)
