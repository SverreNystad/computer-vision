
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

import wandb
from ultralytics import YOLO

if __name__ == "__main__":
    # Load model
    run = wandb.init()
    artifact = run.use_artifact('sverrenystad-ntnu/cv-rgb-small/run_f2boeerl_model:v0', type='model')
    artifact_dir = artifact.download()
    model_name = "best.pt"
    model_path = artifact_dir + "/" + model_name
    model = YOLO(model_path)    

    # Do prediction:
    results = model.predict(
        source=f"/work/{USER_NAME}/computer-vision/data/rgb/images/test",
        project="submissions",
        name="cv-rgb-small",
        save=True,
        save_txt=True,
        save_conf=True
        )
    