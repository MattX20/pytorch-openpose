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

def encode_debug_figures(input_dir: Path, output_path: Optional[Path] = None, fps=10):
    still_frames = sorted(list(input_dir.glob("*.png")))
    logging.info(f"Encoding {len(still_frames)} frames")
    # Define default output path if not provided
    if output_path is None:
        output_path = input_dir / "output.gif"
    assert not output_path.exists(), f"video file already found {output_path}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Create a clip from the images
    clip = ImageSequenceClip([str(img) for img in still_frames], fps=fps)

    
    if output_path.suffix.lower() == ".gif":
        # Write the clip to a GIF file
        clip.write_gif (str(output_path), fps=fps)
    else:
        # Write the clip to a video file MP4/MOV/AVI
        clip.write_videofile(str(output_path), fps=fps)


def main():
    parser = argparse.ArgumentParser(
        description="Run Openpose on a video")
    add_shared_parser_options(parser)
    parser.add_argument("-fps", "--framerate", type=int, default=10)
    args = parser.parse_args()
    input_path = Path(args.input)
    out_dir = args.output_dir
    video_path = None
    assert input_path.exists()
    if input_path.suffix.lower().replace(".", "") in VIDEO_EXT:
        figures_dir = input_path.parent / input_path.stem
        if not out_dir:
            video_path = figures_dir/(input_path.with_suffix(".gif").name) # Name the gif after the mp4 name
    else:
        figures_dir = input_path # Directly providing a debug folder 
    
    if not video_path: #else = forced path
        if out_dir:
            out_dir = Path(out_dir)
            if out_dir.suffix.lower().replace(".", "") in ["gif"] + VIDEO_EXT:
                video_path = out_dir
            else:
                video_path = out_dir/(input_path.with_suffix(".gif").name)  # Name the gif after the mp4 name
        else:
            video_path = None
    # print(figures_dir, video_path)
    encode_debug_figures(figures_dir, video_path, fps=args.framerate)
    



if __name__ == '__main__':
    main()    