import requests
import os
import torch
from effdet import get_efficientdet_config, EfficientDet, DetBenchPredict
from effdet.efficientdet import HeadNet

if __name__ == "__main__":
    # Choose the EfficientDet variant (e.g., 'efficientdet_d0')
    model_name = 'tf_efficientdet_d2'
    config = get_efficientdet_config(model_name)

    # Adjust config parameters as needed (for instance, set the number of classes)
    # For COCO, you might use config.num_classes = 90; for your custom dataset, change accordingly.
    config.num_classes = 2#91  # change to your number of classes (background + objects)

    # Create the model. Use 'pretrained_backbone=True' to initialize with ImageNet weights.
    model = EfficientDet(config, pretrained_backbone=True)

    # Wrap the model for inference with DetBenchPredict for convenience.
    model = DetBenchPredict(model)
    model.eval()  # set the model to evaluation mode

    # Print a summary (optional)
    #print(mode)
    #print(type(model))
