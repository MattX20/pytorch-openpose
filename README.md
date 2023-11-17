## pytorch-openpose

This repository is a fork of the original [pythorch-openpose](https://github.com/Hzzone/pytorch-openpose).

## Model weights

Model weights can be found on this [google drive](https://drive.google.com/drive/folders/1JsvI4M4ZTg98fmnCZLFM-3TeovnCRElG). Save the file `body_pose_model.pth` in a `model` folder inside this directory.


## Video processor

Process a single video at once.

```bash
python3 video_processing.py -i data/0001_pink_ball_vertical_throw.mp4 --trim 5. 6.2
```