from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import FasterRCNN

class SveTimBackbone(nn.Module):
    def __init__(self, out_channels=128):
        super(SveTimBackbone, self).__init__()
        self.out_channels = out_channels
        self.c1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)  # Output: [B, 64, H/2, W/2]
        self.c2 = nn.Conv2d(64, out_channels, kernel_size=3, stride=2, padding=1) # Output: [B, 128, H/4, W/4]

    def forward(self, x: torch.Tensor):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        return OrderedDict([("0", x)])

class SveTimHead(nn.Module):
    def __init__(self, in_channels, representation_size, pool_size):
        super(SveTimHead, self).__init__()
        self.out_channels=representation_size
        self.flattened_size = in_channels * pool_size * pool_size 
        self.fc1 = nn.Linear(self.flattened_size, representation_size)
        self.fc2 = nn.Linear(representation_size, representation_size)
    
    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class SveTimPredictor(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SveTimPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        scores = self.cls_score(x)          # Classification logits for each class.
        bbox_deltas = self.bbox_pred(x)       # Bounding box regressions.
        return scores, bbox_deltas

class SveTimRPNHead(nn.Module):
    """
    A custom RPN head that replaces the default head.
    It uses a shared convolutional layer followed by separate 1x1 convolutions
    for objectness classification (logits) and bounding box regression.
    """
    def __init__(self, in_channels, num_anchors):
        super(SveTimRPNHead, self).__init__()
        # Shared 3x3 convolutional layer.
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # Classification: outputs a single objectness score per anchor.
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1)
        # Regression: outputs 4 values per anchor (dx, dy, dw, dh).
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)
    
    def forward(self, x):
        # x can be a list of feature maps if using FPN.
        logits = []
        bbox_regs = []
        # Process each feature map individually.
        for feature in x:
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_regs.append(self.bbox_pred(t))
        return logits, bbox_regs

def kaiming_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        # Apply Kaiming normal initialization for weights
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def get_custom_detector():
    pool_size = 7

    backbone = SveTimBackbone()
    head = SveTimHead(in_channels=backbone.out_channels, representation_size=backbone.out_channels, pool_size=pool_size)
    predictor = SveTimPredictor(in_channels=head.out_channels, num_classes=2)

    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),  # Tuple of tuples. Here we use one tuple per feature map.
        aspect_ratios=((0.5, 1.0, 2.0),)    # Same number of aspect ratios for each feature map.
    )

    rpn_head = SveTimRPNHead(in_channels=backbone.out_channels, num_anchors=anchor_generator.num_anchors_per_location()[0])

    # Create an ROI pooler.
    # This layer crops the features corresponding to each proposed region into a fixed size,
    # which is then used by the box head to predict final classes and bounding box refinements.
    roi_pooler = ops.MultiScaleRoIAlign(
        featmap_names=['0'],  # Since our backbone returns a single feature map, we refer to it as '0'.
        output_size=pool_size,        # The spatial resolution of the output feature maps after pooling.
        sampling_ratio=2      # A parameter for the pooling operation.
    )

    model = FasterRCNN(
        backbone=backbone,
        box_head=head,
        rpn_head=rpn_head,
        #num_classes=2,
        box_predictor=predictor,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )
    
    model.apply(kaiming_init)

    # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(total_params)
    return model
