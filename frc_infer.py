import os
import torch
from torchvision import transforms
from torchvision.utils import draw_bounding_boxes
from PIL import Image

# ---------------------------
# 1. Configuration
# ---------------------------
# Path to the folder containing input images
INPUT_FOLDER = "../images"
# Path to the folder where output images with drawn boxes will be saved
OUTPUT_FOLDER = "../inferred_images"
# Path to the saved model checkpoint
MODEL_CHECKPOINT = "./cv-svetimdet-final/checkpoint_3000.pt"
# Score threshold: keep detections with score >= SCORE_THRESH
SCORE_THRESH = 0.5

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------------------------
# 2. Load the model
# ---------------------------
# weights_only=False ensures we load the full model (arch + weights)
model = torch.load(MODEL_CHECKPOINT, weights_only=False)
model.eval()  # set to evaluation mode

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ---------------------------
# 3. Prepare the image transform
# ---------------------------
# Torch expects images as C×H×W tensors in [0,1] range
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts PIL Image or numpy.ndarray to FloatTensor in [0,1]
])

# ---------------------------
# 4. Inference loop
# ---------------------------
for fname in os.listdir(INPUT_FOLDER):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue  # skip non-image files

    # 4.1 Load and transform image
    img_path = os.path.join(INPUT_FOLDER, fname)
    pil_img = Image.open(img_path).convert("RGB")
    img_tensor = transform(pil_img).to(device)             # shape: [3, H, W]

    # 4.2 Run the model (returns list of dicts; batch size = 1 here)
    with torch.no_grad():
        outputs = model([img_tensor])

    # outputs is a list; take the first (and only) element
    output = outputs[0]
    boxes  = output["boxes"]    # Tensor of shape [N, 4], in (xmin, ymin, xmax, ymax) format
    scores = output["scores"]   # Tensor of shape [N]
    labels = output["labels"]   # Tensor of shape [N]

    # 4.3 Filter detections by score threshold
    keep = scores >= SCORE_THRESH
    boxes  = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]  # in case you want to draw class labels too

    # If no boxes remain, you can skip drawing or save the original
    if boxes.numel() == 0:
        # Optionally save the original unchanged image
        pil_img.save(os.path.join(OUTPUT_FOLDER, fname))
        continue

    # 4.4 Draw boxes onto the image
    # draw_bounding_boxes expects a byte tensor (0–255) of dtype=torch.uint8
    img_uint8 = (img_tensor * 255).to(torch.uint8)
    # Choose a color per box (here all red)
    box_img = draw_bounding_boxes(
        image=img_uint8,
        boxes=boxes.cpu(),
        labels=[f"{s:.2f}" for s in scores.cpu()],  # draw scores as captions
        colors="red",
        width=2,
        font_size=16,
    )

    # 4.5 Convert back to PIL and save
    # torch tensor shape is [3, H, W], convert to [H, W, 3]
    np_img = box_img.permute(1, 2, 0).cpu().numpy()
    out_pil = Image.fromarray(np_img)
    out_pil.save(os.path.join(OUTPUT_FOLDER, fname))

print("Done! All images processed.")

