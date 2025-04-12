import os
import getpass
import logging
import sys
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
from optuna.study import StudyDirection
from ultralytics import YOLO
from codecarbon import track_emissions, OfflineEmissionsTracker
from dotenv import load_dotenv
import os

load_dotenv()

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


def objective(trial: optuna.Trial):
    epochs = trial.suggest_int("epochs", 1, 6000)

    # From the YOLO default search space
    lr0 = trial.suggest_float("lr0", 1e-5, 1e-1)
    lrf = trial.suggest_float("lrf", 0.01, 1.0)
    momentum = trial.suggest_float("momentum", 0.6, 0.98)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.001)
    warmup_epochs = trial.suggest_float("warmup_epochs", 0.0, 5.0)
    warmup_momentum = trial.suggest_float("warmup_momentum", 0.0, 0.95)
    hsv_h = trial.suggest_float("hsv_h", 0.0, 1.0)
    hsv_s = trial.suggest_float("hsv_s", 0.0, 1.0)
    hsv_v = trial.suggest_float("hsv_v", 0.0, 1.0)
    degrees = trial.suggest_float("degrees", 0.0, 45.0)
    translate = trial.suggest_float("translate", 0.0, 1.0)
    scale = trial.suggest_float("scale", 0.0, 0.9)
    shear = trial.suggest_float("shear", 0.0, 10.0)
    flipud = trial.suggest_float("flipud", 0.2, 0.8)
    fliplr = 0.5 
    perspective = 0.00012

    # Initialize model
    model = YOLO("yolo11s.yaml")

    # Train model with the sampled hyperparameters
    model.train(
        data=f"{DEVICE_PATH}/{USER_NAME}/computer-vision/data/data-combined.yaml",
        project="cv-combined-small",
        optimizer="AdamW",
        epochs=epochs,
        imgsz=1024,
        lr0=lr0,
        lrf=lrf,
        momentum=momentum,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        warmup_momentum=warmup_momentum,
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
    )

    # Validate and retrieve metrics
    metrics = model.val()
    precision, recall, mAP50, mAP50_95, fitness = metrics.results_dict.values()
    wandb.log({
        "precision": precision, "recall": recall, "mAP50": mAP50, "mAP50_95": mAP50_95, "fitness": fitness
    })
    return mAP50_95


@track_emissions(offline=True, country_iso_code="NOR")
def main(study_name: str):
    tracker = OfflineEmissionsTracker(country_iso_code="NOR")
    try:
        tracker.start()
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        wandb_kwargs = {"project": study_name}
        wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs, as_multirun=True)

        MYSQL_USER = os.getenv("MYSQL_USER")
        MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")

        storage_name = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@mysql.stud.ntnu.no/timma_tdt4265_db"

        directions = [
            StudyDirection.MAXIMIZE,  # mAP50_95
        ]

        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            load_if_exists=True,
            directions=directions,
        )

        # Run your study
        for _ in range(100):
            study.optimize(objective, n_trials=10, callbacks=[wandbc])
            tracker.flush()
    finally:
        tracker.stop()


if __name__ == "__main__":
   study_name = "cv-combined-small"
   main(study_name)
