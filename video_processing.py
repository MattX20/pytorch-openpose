from pathlib import Path
import argparse

from moviepy.editor import VideoFileClip
import numpy as np
from main import main as main_processing, get_model
from cv2 import resize, rotate, ROTATE_90_CLOCKWISE, ROTATE_90_COUNTERCLOCKWISE, ROTATE_180
from typing import List, Union, Optional, Tuple
import logging
from shared import add_shared_parser_options, add_video_parser_args, add_visualizer_parser_args, get_trim
from visualize_results import encode_debug_figures

def process_video_frames(
        video_path: Path,
        visualization_dir:Path,
        trim: Optional[Tuple[Union[int, None], Union[int, None]]]=None,
        rotation=None,
        model=None,
    ) -> List[np.ndarray]:
    """Run open pose (decode video using MoviePy)

    Args:
        video_path (Path):  Path to the video file.

    """
    # @TODO: skip frames until start without decoding (directly in moviepy?)
    # @TODO: export pose estimation debug videos.
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

def main():
    parser = argparse.ArgumentParser(
        description="Run Openpose on a video")
    add_shared_parser_options(parser)
    add_video_parser_args(parser)
    add_visualizer_parser_args(parser)
    args = parser.parse_args()
    video_path = Path(args.input)
    assert video_path.exists()
    out_dir = args.output_dir
    if not out_dir:
        out_dir = video_path.parent / video_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
    trim = get_trim(args)
    process_video_frames(video_path, visualization_dir=out_dir, trim=trim, rotation=None)
    if args.visualize:
        encode_debug_figures(out_dir, out_dir/(video_path.with_suffix(".gif").name))



if __name__ == '__main__':
    main()    