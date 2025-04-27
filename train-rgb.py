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
from codecarbon import track_emissions, OfflineEmissionsTracker
import wandb
from dotenv import load_dotenv
import os

load_dotenv()


WANDB_API_KEY = os.getenv("WANDB_API_KEY")
print(f"WANDB_API_KEY: {WANDB_API_KEY}")
# Initialize your Weights & Biases environment
wandb.login(key=WANDB_API_KEY)

def objective(trial: optuna.Trial):

    # From the YOLO default search space
    hsv_h = trial.suggest_float("hsv_h", 0.015, 0.03)
    hsv_s = trial.suggest_float("hsv_s", 0.5, 0.7)
    hsv_v = trial.suggest_float("hsv_v", 0.3, 0.5)
    degrees = trial.suggest_float("degrees", 5.0, 15.0)
    translate = trial.suggest_float("translate", 0.0, 0.2)
    scale = trial.suggest_float("scale", 0.3, 0.6)
    shear = trial.suggest_float("shear", 0.0, 5.0)
    perspective = trial.suggest_float("perspective", 0.0, 0.001)
    fliplr = 0.5
    mosaic = trial.suggest_float("mosaic", 0.0, 1.0)
    mixup = trial.suggest_float("mixup", 0.0, 0.2)
    copy_paste = trial.suggest_float("copy_paste", 0.0, 1.0)

    # Initialize model
    model = YOLO("yolo11s.yaml")

    # Train model with the sampled hyperparameters
    model.train(
        data=f"/work/{USER_NAME}/computer-vision/data/data-rgb.yaml",
        project="cv-rgb-small-resize",
        epochs=70,
        hsv_h=hsv_h,
        hsv_s=hsv_s,
        hsv_v=hsv_v,
        degrees=degrees,
        translate=translate,
        scale=scale,
        shear=shear,
        perspective=perspective,
        fliplr=fliplr,
        mosaic=mosaic,
        mixup=mixup,
        imgsz=1920,
    )

    # Validate and retrieve metrics
    metrics = model.val()
    precision, recall, mAP50, mAP50_95, fitness = metrics.results_dict.values()

    return precision, recall, mAP50, mAP50_95, fitness


@track_emissions(offline=True, country_iso_code="NOR")
def main(study_name: str):
    tracker = OfflineEmissionsTracker(country_iso_code="NOR")
    try:
        tracker.start()
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

        MYSQL_USER = os.getenv("MYSQL_USER")
        MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")

        storage_name = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@mysql.stud.ntnu.no/timma_tdt4265_db"

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

        # Run your study (e.g. 10 iterations)
        for _ in range(100):
            study.optimize(objective, n_trials=10)
            tracker.flush()
    finally:
        emmisions = tracker.stop()

if __name__ == "__main__":
   study_name = "rgb_yolo_epochs"
   main(study_name)
   # run_server(study_name)
