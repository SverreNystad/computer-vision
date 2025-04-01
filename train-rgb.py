import os
import getpass

USER_NAME = getpass.getuser()

os.environ["WANDB_DIR"] = f"/work/{USER_NAME}/"
os.environ["WANDB_CACHE_DIR"] = f"/work/{USER_NAME}/.cache/"
os.environ["WANDB_CONFIG_DIR"] = f"/work/{USER_NAME}/.config/wandb"
os.environ["WANDB_DATA_DIR"] = f"/work/{USER_NAME}/.cache/wandb-data/"
os.environ["WANDB_ARTIFACT_DIR"] = f"/work/{USER_NAME}/artifacts"

if not os.path.exists(os.getenv("WANDB_CONFIG_DIR")):
    os.makedirs(os.getenv("WANDB_CONFIG_DIR"))

import logging
import sys
import optuna
from optuna.study import StudyDirection
from optuna_dashboard import run_server
from ultralytics import YOLO
from codecarbon import track_emissions
import wandb
from dotenv import load_dotenv
import os

load_dotenv()


WANDB_API_KEY = os.getenv("WANDB_API_KEY")
print(f"WANDB_API_KEY: {WANDB_API_KEY}")
# Initialize your Weights & Biases environment
wandb.login(key=WANDB_API_KEY)

def objective(trial: optuna.Trial):
    epochs = trial.suggest_int("epochs", 1, 10)

    # From the YOLO default search space
    lr0 = trial.suggest_float("lr0", 1e-5, 1e-1, log=True)
    lrf = trial.suggest_float("lrf", 0.01, 1.0)
    momentum = trial.suggest_float("momentum", 0.6, 0.98)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.001)
    warmup_epochs = trial.suggest_float("warmup_epochs", 0.0, 5.0)
    warmup_momentum = trial.suggest_float("warmup_momentum", 0.0, 0.95)
    box = trial.suggest_float("box", 0.02, 0.2)
    cls = trial.suggest_float("cls", 0.2, 4.0)
    hsv_h = trial.suggest_float("hsv_h", 0.0, 0.1)
    hsv_s = trial.suggest_float("hsv_s", 0.0, 0.9)
    hsv_v = trial.suggest_float("hsv_v", 0.0, 0.9)
    degrees = trial.suggest_float("degrees", 0.0, 45.0)
    translate = trial.suggest_float("translate", 0.0, 0.9)
    scale = trial.suggest_float("scale", 0.0, 0.9)
    shear = trial.suggest_float("shear", 0.0, 10.0)
    perspective = trial.suggest_float("perspective", 0.0, 0.001)
    flipud = trial.suggest_float("flipud", 0.0, 1.0)
    fliplr = trial.suggest_float("fliplr", 0.0, 1.0)
    mosaic = trial.suggest_float("mosaic", 0.0, 1.0)
    mixup = trial.suggest_float("mixup", 0.0, 1.0)
    copy_paste = trial.suggest_float("copy_paste", 0.0, 1.0)

    # Initialize model
    model = YOLO("yolo11n.yaml")

    # Train model with the sampled hyperparameters
    model.train(
        data=f"/work/{USER_NAME}/computer-vision/data/data.yaml",
        project="cv-rgb",
        epochs=epochs,
        lr0=lr0,
        lrf=lrf,
        momentum=momentum,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        warmup_momentum=warmup_momentum,
        box=box,
        cls=cls,
        hsv_h=hsv_h,
        hsv_s=hsv_s,
        hsv_v=hsv_v,
        degrees=degrees,
        translate=translate,
        scale=scale,
        shear=shear,
        perspective=perspective,
        flipud=flipud,
        fliplr=fliplr,
        mosaic=mosaic,
        mixup=mixup,
        copy_paste=copy_paste,
    )

    # Validate and retrieve metrics
    metrics = model.val()
    precision, recall, mAP50, mAP50_95, fitness = metrics.results_dict.values()

    return precision, recall, mAP50, mAP50_95, fitness


@track_emissions(offline=True, country_iso_code="NOR")
def main(study_name: str):
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    POSTGRES_USER = os.getenv("POSTGRES_USER")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
    storage_name = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@10.22.130.139:5432/computer_vision_db"
    #storage_name = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@optuna.techtaitans.com:5432/computer_vision_db"



    directions = [
        StudyDirection.MAXIMIZE,  # precision
        StudyDirection.MAXIMIZE,  # recall
        StudyDirection.MAXIMIZE,  # mAP50
        StudyDirection.MAXIMIZE,  # mAP50-95
        StudyDirection.MAXIMIZE,  # fitness
    ]

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        directions=directions,
    )

    # Run your study
    for _ in range(10):
        study.optimize(objective, n_trials=10)


if __name__ == "__main__":
   study_name = "lidar_yolo_epochs"
   main(study_name)
   run_server(study_name)
