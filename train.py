from ultralytics import YOLO
import torch
from torchvision.transforms import v2
import optuna

import optunahub
from optuna.study import StudyDirection
from optuna_dashboard import run_server
import logging
import sys


def objective(trial: optuna.Trial) -> float:


    epochs = trial.suggest_int("epochs", 1, 3_000)


    model = YOLO("yolo11n.yaml")
    model.train(
        data="/work/sverrnys/computer-vision/data/data.yaml",
        epochs=epochs,
        # degrees=0.25,    # rotation in degrees (+/-)
        # translate=0.1,   # translation fraction
        # scale=0.3,       # scaling factor gain
        # shear=0.0,       # shear (set to 0.0 to disable)
        # # flipud=0.0,      # probability for vertical flip
        # fliplr=0.5,      # probability for horizontal flip
        # mosaic=1.0,      # mosaic augmentation (combines 4 images)
        # mixup=0.0,       # mixup augmentation probability (set to 0 to disable)
        # copy_paste=0.0   # copy-paste augmentation probability
        )

    metrics = model.val()
    precision, recall, mAP50, mAP50_95, fitness = metrics.results_dict.vaules()
    
    # results_dict: {'metrics/precision(B)': np.float64(0.21608179959939197), 
    # 'metrics/recall(B)': np.float64(0.10619469026548672), 
    # 'metrics/mAP50(B)': np.float64(0.04932460480212709), 
    # 'metrics/mAP50-95(B)': np.float64(0.00941862329112571), 
    # 'fitness': np.float64(0.013409221442225849)}

    return precision, recall, mAP50, mAP50_95, fitness
    

if __name__ == "__main__":
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "yolo_epochs" 
    storage_name = f"sqlite:///{study_name}.db"
    directions = [
        StudyDirection.MAXIMIZE, 
        StudyDirection.MAXIMIZE, 
        StudyDirection.MAXIMIZE, 
        StudyDirection.MAXIMIZE, 
        StudyDirection.MAXIMIZE, 
        ]


    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists= True,
        directions=directions
        )


    for _ in range(10):
         study.optimize(objective, n_trials=1)
    run_server(storage_name)
