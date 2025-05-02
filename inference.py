import torch
import time
import timeit
from PIL import Image
import numpy as np

model = torch.load("./cv-svetimdet-final/checkpoint_3000.pt", weights_only=False)

input_data = torch.randn(1, 3, 1024, 1024).to(torch.device('cuda'))
img = Image.open("./data/rgb/images/test/frame_000005.PNG")
img = np.array(img, dtype=np.uint8)
input_data = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).to(torch.device('cuda')).unsqueeze(0)

model.eval()

def infer():
    model.forward(input_data)

print(timeit.timeit("infer()", globals=locals(), number=1_000))
