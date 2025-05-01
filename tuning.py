from ultralytics import YOLO
import os
import getpass
from dotenv import load_dotenv

load_dotenv()

# Load the small YOLOv11 model
model = YOLO("yolo11s.yaml")

USER_NAME = getpass.getuser()

DEVICE_PATH = os.getenv("DEVICE_PATH") or "/work"
os.environ["WANDB_DIR"] = f"{DEVICE_PATH}/{USER_NAME}/"
os.environ["WANDB_CACHE_DIR"] = f"{DEVICE_PATH}/{USER_NAME}/.cache/"
os.environ["WANDB_CONFIG_DIR"] = f"{DEVICE_PATH}/{USER_NAME}/.config/wandb"
os.environ["WANDB_DATA_DIR"] = f"{DEVICE_PATH}/{USER_NAME}/.cache/wandb-data/"
os.environ["WANDB_ARTIFACT_DIR"] = f"{DEVICE_PATH}/{USER_NAME}/artifacts"

if not os.path.exists(os.getenv("WANDB_CONFIG_DIR")):
    os.makedirs(os.getenv("WANDB_CONFIG_DIR"))

import wandb


WANDB_API_KEY = os.getenv("WANDB_API_KEY")
print(f"WANDB_API_KEY: {WANDB_API_KEY}")
# Initialize your Weights & Biases environment
wandb.login(key=WANDB_API_KEY)

DEVICE_PATH = os.getenv("DEVICE_PATH") or "/work"
# Define hyperparameter search space for augmentation and training parameters
search_space = {
    "hsv_h": (0.015, 0.03),  # Hue shift ±1.5–3%
    "hsv_s": (0.55, 0.8),  # Saturation jitter ±50–70%
    "hsv_v": (0.4, 0.7),  # Brightness jitter ±30–50%
    "degrees": (5.0, 15.0),  # Rotation ±5–15°
    "translate": (0.1, 0.2),  # Shift ±10–20%
    "scale": (0.3, 0.6),  # Scale 30–60%
    "shear": (0.0, 5.0),  # Shear 0–5°
    "perspective": (0.0, 0.001),  # Perspective warp up to 0.001
    "fliplr": (0.5, 0.5),  # Horizontal flip fixed at 50%
    "flipud": (0.0, 0.0),  # Vertical flip disabled
    "mosaic": (0.8, 1.0),  # Mosaic on 80–100% of images
    "mixup": (0.0, 0.2),  # Mixup probability 0–20%
}

# Run hyperparameter tuning (genetic algorithm) for augmented YOLO training
results = model.tune(
    data=f"{DEVICE_PATH}/{USER_NAME}/computer-vision/data/data-rgb.yaml",
    epochs=50,  # number of training epochs per trial
    iterations=200,  # number of hyperparameter samples/trials
    optimizer="AdamW",  # optimizer to use during tuning
    space=search_space,  # custom search space defined above
    plots=True,  # generate loss/metric plots
    save=True,  # save best hyperparameters
    val=True,  # run validation every epoch
    project="runs/tune-snow",  # where to store tuning runs
    resume=True,
    imgsz=1920,
)

print("Tuning complete. Best hyperparameters:\n", results)
