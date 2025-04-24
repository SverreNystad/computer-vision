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

from ultralytics import YOLO
import wandb
from dotenv import load_dotenv
import os

load_dotenv()


WANDB_API_KEY = os.getenv("WANDB_API_KEY")
print(f"WANDB_API_KEY: {WANDB_API_KEY}")
# Initialize your Weights & Biases environment
wandb.login(key=WANDB_API_KEY)

# Initialize model
model = YOLO("yolo11s.yaml")

# Train model with the sampled hyperparameters
model.train(
    data=f"/work/{USER_NAME}/computer-vision/data/data-rgb.yaml",
    project="cv-rgb-small-baseline",
    
)

# Validate and retrieve metrics
metrics = model.val()
precision, recall, mAP50, mAP50_95, fitness = metrics.results_dict.values()

print(f"precision: {precision}, recall: {recall}, mAP50: {mAP50}, mAP50_95: {mAP50_95}, fitness: {fitness}")
# results = model.predict(
#         source=f"/work/{USER_NAME}/computer-vision/data/rgb/images/test",
#         project="submissions",
#         name="cv-rgb-small-tune",
#         save_txt=True,
#         save_conf=True
#         )




