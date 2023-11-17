from pathlib import Path
import argparse

from moviepy.editor import VideoFileClip
import numpy as np
from main import main as main_processing, get_model
from cv2 import resize, rotate, ROTATE_90_CLOCKWISE, ROTATE_90_COUNTERCLOCKWISE, ROTATE_180
from typing import List, Union, Optional, Tuple
import logging

def load_video_frames(
        video_path: Path,
        visualization_dir:Path,
        trim: Optional[Tuple[Union[int, None], Union[int, None]]]=None,
        rotation=None
    ) -> List[np.ndarray]:
    """Run open pose (decode video using MoviePy)

    Args:
        video_path (Path):  Path to the video file.

    """
    # @TODO: skip frames until start without decoding (directly in moviepy?)
    # @TODO: export pose estimation debug videos.
    model = None
    poses = []
    with VideoFileClip(str(video_path)) as video:
        if video.rotation in (90, 270): # Support vertical videos
            # https://github.com/Zulko/moviepy/issues/586
            video = video.resize(video.size[::-1])
            video.rotation = 0
        start, end = None, None
        if trim is not None:
            assert len(trim) == 2
            start, end = trim
            if start is not None:
                start = int(start*video.fps)
            if end is not None:
                end = int(end*video.fps)
        for frame_idx, frame in enumerate(video.iter_frames()):
            if end is not None and frame_idx>end:
                logging.info(f"LAST FRAME REACHED! {frame_idx}>{end}")
                break
            if start is not None and frame_idx<= start:
                continue
            if rotation is not None:
                assert rotation in [ROTATE_90_CLOCKWISE, ROTATE_90_COUNTERCLOCKWISE, ROTATE_180]
                frame=rotate(frame, rotateCode=rotation)
            frame = resize(frame, (0, 0), fx=0.2, fy=0.2)
            logging.info(f"processing frame ={frame_idx:04d} | {frame.shape[0]} x {frame.shape[1]}")
            if model is None:
                model = get_model() #Load the model when needed.
            pose = main_processing(
                [frame],
                visualization_dir,
                body_estimation=model,
                image_names=[f"{frame_idx:04d}",],
                save_path=visualization_dir
            )
            poses.append(pose[0, ...])
    return poses


def add_shared_parser_options(parser):
    path_args = parser.add_argument_group("paths")
    path_args.add_argument("-i", "--input", default="data/0001_pink_ball_vertical_throw.mp4", type=str)
    path_args.add_argument("-o", "--output-dir", required=False, type=str)

def main():
    parser = argparse.ArgumentParser(
        description="Run Openpose on a video")
    add_shared_parser_options(parser)
    video_args = parser.add_argument_group("video")
    video_args.add_argument("-t", "--trim", nargs="+", type=float, help="Trim in seconds like -t 4.8 5.3 or -t 0.5")
    args = parser.parse_args()
    video_path = Path(args.input)
    assert video_path.exists()
    out_dir = args.output_dir
    if not out_dir:
        out_dir = video_path.parent / video_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
    trim = args.trim
    if trim:
        if len(trim)==1:
            trim = (None,  trim[0])
        else:
            assert len(trim)==2, "trim shall have one or two elements"
        print(f"Video trimming: {trim}")
    load_video_frames(video_path, visualization_dir=out_dir, trim=trim, rotation=None)
    



if __name__ == '__main__':
    main()    