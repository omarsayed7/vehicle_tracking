# Vehicle Tracking using YOLOv8 and DeepSORT

In this project, we use the power of YOLOv8 to accurately detect vehicles in images and videos. But that's not all—DeepSORT, a cutting-edge framework, takes it a step further by enabling vehicle tracking across multiple frames!


# Installation

To use this project, first clone the repo on your device using the command below:

Clone the repo

```
https://github.com/omarsayed7/vehicle_tracking.git
```

Create new conda environment containing the dependencies in the [requirements file](requirements.yشml) .

```
#If using pip
pip install -r requirements.txt

#If using conda
conda env create -f requirements.yaml
```

Activate the created environment.

```
conda activate vehicle_tracking
```


# Project Structure:

```
├── README.md
├── .gitignore
├── .gitmodules
├── requirements.txt
├── requirements.yaml
├── data
|   └── road_1.mp4
└── src
    ├── config
    │   └── celebrity_recognition_inference.ipynb
    ├── deep_sort_pytorch @ a050050
    ├── predictor.py
    ├── utils.py
    └── run.py

```


## Usage

before runing the script, make sure you place the required video file to process in `data` directory.
```Python
HYDRA_FULL_ERROR=1 python run.py source="../data/road_1.mp4" model="yolov8x.pt"

--model           yolov8 model size
--source          relative path of the video file
```
## Results Samples
https://drive.google.com/drive/folders/1K-Vl_T3hGlbNtGj4VwhO_52q0JfgxZET?usp=sharing
