# SnowPole Object Detection with LiDAR data and RGB
There has been major advancements in Autonomous driving in recent years, attributed to computer
vision (DL). However, such models are mostly focused on driving in ideal conditions, and thus struggles
with challenging conditions like snowy roads. One way to localize the road in winter time is by relying
on the location of snow poles, which are typically erected on either side of the road in areas prone to
snow in the winter. Our task here is to perform real time object detection of snow poles, in order to
further develop AD capabilities in winter conditions.

Our goal was the following:

_We wanted to develop a lightweight model that can reliably spot snow poles before the car needs them, while fitting the power/compute budget of an edge device. Meeting this challenge pushes the winter capability of autonomous driving a crucial step closer to production in the Nordics._

## Dataset
The dataset was collected by the [NAPLab](https://www.ntnu.edu/idi/naplab) at NTNU.

We had access to two separate datasets for this task. The first dataset is a selection of natural images
(RGB), and the second dataset consists of LiDAR images. The LiDAR images are combined as
RGB images by combining Near-IR, Signal, and Reflectivity channels. Near-IR maps to blue,
Signal to green, and Reflectivity to red. 
As redistribution of the dataset is prohibited the images and labels are not in the repository. To reproduce the findings or use the data contact [NAPLab](https://www.ntnu.edu/idi/naplab) at NTNU.

## Our results
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/f30d3d53-7b4e-4ef5-bd1a-94baa69ae34b" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/9205e7e7-a23f-4aff-88ee-eb928fdefab9" />



## Usage

Train model on rgb dataset run the following command:
```bash
python train-rgb.py
```

Train model on lidar dataset run the following command:
```bash
python train-lidar.py
```

Train model on combined dataset run the following command:
```bash
python train-combined.py
```

### Submission

To create a submission run the following command:
```bash
python submission.py
```
