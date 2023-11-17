from pathlib import Path
import argparse

from moviepy.editor import VideoFileClip
import numpy as np
from video_processing import add_shared_parser_options
from typing import Optional
from moviepy.editor import ImageSequenceClip
import moviepy.editor as mp
import logging
VIDEO_EXT = ["mp4", "avi", "mp4", "mov"]

def encode_debug_figures(input_dir: Path, output_path: Optional[Path] = None):
    still_frames = sorted(list(input_dir.glob("*.png")))
    logging.info(f"Encoding {len(still_frames)} frames")
    # Define default output path if not provided
    if output_path is None:
        output_path = input_dir / "output.gif"

    # Create a clip from the images
    clip = ImageSequenceClip([str(img) for img in still_frames], fps=10)

    # Write the clip to a GIF file
    clip.write_gif(str(output_path), fps=10)


def main():
    parser = argparse.ArgumentParser(
        description="Run Openpose on a video")
    add_shared_parser_options(parser)
    args = parser.parse_args()
    video_path = Path(args.input)
    out_dir = args.output_dir
    assert video_path.exists()
    if video_path.suffix.lower().replace(".", "") in VIDEO_EXT:
        figures_dir = video_path.parent / video_path.stem
        figures_dir.mkdir(parents=True, exist_ok=True)
        if not out_dir:
            video_path = figures_dir/(video_path.with_suffix(".gif").name) # Name the gif after the mp4 name
    else:
        figures_dir = video_path # Directly providing a debug folder 

    if video_path is None:
        if out_dir:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            if out_dir.suffix.lower().replace(".", "") in ["gif"] + VIDEO_EXT:
                video_path = out_dir
            else:
                video_path = out_dir/(video_path.with_suffix(".gif").name)  # Name the gif after the mp4 name
        else:
            video_path = None
    encode_debug_figures(figures_dir, video_path)
    



if __name__ == '__main__':
    main()    