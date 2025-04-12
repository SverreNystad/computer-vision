import requests
import os
import torch
import torch.optim as optim
from effdet import get_efficientdet_config, EfficientDet, DetBenchPredict
from effdet.efficientdet import HeadNet
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import getpass
from tqdm import tqdm

USER_NAME = getpass.getuser()

class OurDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.img_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith((".jpg", ".png"))])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, i):
        img_file = self.img_files[i]
        img_path = os.path.join(self.img_dir, img_file)

        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(self.label_dir, label_file)

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        boxes = []
        labels = []

        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
            
                id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])
                
                boxes.append([x_center, y_center, width, height])
                labels.append(id)

        targets = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.empty((0,), dtype=torch.float32)
        }

        return img, targets

def objective(trail: optuna.Trail):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_img_dir = f"/work/{USER_NAME}/computer-vision/data/rgb/train/images"
    train_label_dir = f"/work/{USER_NAME}/computer-vision/data/rgb/train/labels"
    val_img_dir = f"/work/{USER_NAME}/computer-vision/data/rgb/val/images"
    val_label_dir = f"/work/{USER_NAME}/computer-vision/data/rgb/val/labels"

    train_dataset = OurDataset(train_img_dir, train_label_dir, transform=transform)
    val_dataset = OurDataset(val_img_dir, val_label_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=2)

    # Choose the EfficientDet variant (e.g., 'efficientdet_d0')
    model_name = 'tf_efficientdet_d2'
    config = get_efficientdet_config(model_name)

    # Adjust config parameters as needed (for instance, set the number of classes)
    # For COCO, you might use config.num_classes = 90; for your custom dataset, change accordingly.
    config.num_classes = 1#91  # change to your number of classes (background + objects)

    # Create the model. Use 'pretrained_backbone=True' to initialize with ImageNet weights.
    model = EfficientDet(config, pretrained_backbone=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    num_epochs = trail.suggest_int("epochs", 1, 6000)
    learning_rate = trail.suggest_float("lr", 1e-5, 1e-1)
    weight_decay = trail.suggest_float("weight_decay", 0.0, 0.001)
    

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in tqdm(range(num_epochs)):
        model.train()
        epoch_loss = 0.0

        for images, targets in train_loader:
            images = [image.to_device(device) for image in images]
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

            optimizer.zero_grad()

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())

            losses.backward()

            optimizer.step()

            epoch_loss += losses.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] completed. Average Training Loss: {avg_epoch_loss:.4f}")

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, targets in val_loader:
                images = [image.to_device(device) for image in images]
                targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()

        avg_val_loss = val_loss / len(val_loader)

        

    # # Wrap the model for inference with DetBenchPredict for convenience.
    # model = DetBenchPredict(model)
    # model.eval()  # set the model to evaluation mode


# if __name__ == "__main__":
#     

