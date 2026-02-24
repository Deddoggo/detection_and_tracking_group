## Report for all in progress
## Quick install:
### 1. Create an enviroment.:
```
conda create -n tracking_env python=3.8
conda activate tracking_env
```
Newer version:
```
conda create -n yolo_env python=3.10
conda activate yolo_env
```
### 2. Install Dependencies
```
pip install -r requirements.txt
```
```
pip install git+https://github.com/facebookresearch/detectron2.git@536dc9d527074e3b15df5f6677ffe1f4e104a4ab
```
Newer version:
```
pip install ultralytics boxmot transformers accelerate timm
pip install opencv-python numpy scikit-learn
```
### 3. Downloads folder resources and move it in folder deep_sort, you can ignore folder detection and only download networks folder:
https://drive.google.com/drive/folders/18fKzfqnqhqW3s9zwsCbnVJ5XF2JFeqMp
Download the weights folder for newer yolo, reid, mixste models. Extract the downloaded files and place them inside root of this project:
https://drive.google.com/drive/folders/1KUbfE7mWMYmqWQwFIHN_WlORQy9Dq_cg?usp=sharing
Your directory structure should look like this:
detection_and_tracking_group/
├── yolo12x.pt
├── osnet_ain_x1_0_msmt17.pt
├── best_epoch_cpn_243f.bin
└── ...
├── yolo_detection_vitpose_draw.py
├── identify_group_social.py
└── README.md
### 4. Run below cmd line to detection human and create folder frames:
```
python detection.py --input_video [Link to your video] --output_folder [Link to your output folder]
```
### 5. Run below cmd line to tracking people and identify groups:
```
python tracking_group.py --input_video [Link to your video] --frames_dir [Link to your output folder, same above output folder] --epsilon [DBSCAN epsilon sush as 50, 75, 100] --threshold_overlap [threshold overlap with old group (0.5 -> 1.0)]
```
### 6. You can see the results in folder output

## Citation / References
```bibtex
@inproceedings{Wojke2017simple,
  title={Simple Online and Realtime Tracking with a Deep Association Metric},
  author={Wojke, Nicolai and Bewley, Alex and Paulus, Dietrich},
  booktitle={2017 IEEE International Conference on Image Processing (ICIP)},
  year={2017},
  pages={3645--3649},
  organization={IEEE},
  doi={10.1109/ICIP.2017.8296962}
}

@inproceedings{Wojke2018deep,
  title={Deep Cosine Metric Learning for Person Re-identification},
  author={Wojke, Nicolai and Bewley, Alex},
  booktitle={2018 IEEE Winter Conference on Applications of Computer Vision (WACV)},
  year={2018},
  pages={748--756},
  organization={IEEE},
  doi={10.1109/WACV.2018.00087}
}

@misc{wu2019detectron2,
  author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                  Wan-Yen Lo and Ross Girshick},
  title =        {Detectron2},
  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
  year =         {2019}
}

@article{article,
author = {Khan, Muiz and Paul, Pias and Rashid, Mahmudur and Hossain, Mainul and Ahad, Md Atiqur Rahman},
year = {2020},
month = {10},
pages = {507 - 517},
title = {An AI-Based Visual Aid With Integrated Reading Assistant for the Completely Blind},
volume = {50},
journal = {IEEE Transactions on Human-Machine Systems},
doi = {10.1109/THMS.2020.3027534}
}

