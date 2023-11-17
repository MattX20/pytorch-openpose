## pytorch-openpose

This repository is a fork of the original [pythorch-openpose](https://github.com/Hzzone/pytorch-openpose).

## Model weights

Model weights can be found on this [google drive](https://drive.google.com/drive/folders/1JsvI4M4ZTg98fmnCZLFM-3TeovnCRElG). Save the file `body_pose_model.pth` in a `model` folder inside this directory.


## Video processing

### Batch processor
[Learn more about `batch-processing`](https://github.com/emmcb/batch-processing)

```bash
python3 batch.py -i 'data/*.mp4' -o _openpose --trim 0.5 -v --skip-existing --visualization-suffix mp4
```

- `-i` path to videos, use a regex or provide a list to .mp4
- `-o` output directory
- `--trim` allows selecting a segment in the video
- `-v` to visualize a gif or mp4.
- `-vsuf` mp4 or gif
- `-fps` 10 ... use to visualize results slowly.
- `-mp` :warning: **Multi-processing** 
  - use cautiously 
  - `-j 2` specifies the number of threads
  - `python3 batch.py -i 'data/*.mp4' -o _openpose --trim 0.5 -v --skip-existing --visualization-suffix mp4 -mp -j 4`


### Single video processor

Process a single video at once.
- `-i` path to input video (.mp4)
- `--trim` allows selecting a segment in the video
- `-v` to visualize a gif.
```bash
python3 video_processing.py -i data/0001_pink_ball_vertical_throw.mp4 --trim 5.8 6.2 -v
```