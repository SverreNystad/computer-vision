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
    yolo_model_types = ["yolo11s.yaml", "yolo11n.yaml"]

    # Basic parameters
    epochs = trial.suggest_int("epochs", 200, 3000)
    imgsz = trial.suggest_int("imgsz", 640, 1344)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    model_type = trial.suggest_categorical("model_type", choices=yolo_model_types)
    # Initialize model
    model = YOLO(model_type)

    # Train model with the sampled hyperparameters
    # Run hyperparameter tuning (genetic algorithm) for augmented YOLO training
    results = model.train(
        data=f"/work/{USER_NAME}/computer-vision/data/data-combined.yaml",
        epochs=epochs,  # number of training epochs per trial
        project="cv-combined-albuations",
        imgsz=imgsz,
        patience=200,
        dropout=dropout,
        mosaic=0,
        lr0=0.0015,
        optimizer="auto",  # optimizer to use during tuning
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
   study_name = "lidar_yolo_epochs"
   main(study_name)
   # run_server(study_name)
