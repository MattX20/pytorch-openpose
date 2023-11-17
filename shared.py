import argparse
from typing import Tuple
VIDEO_EXT = ["mp4", "avi", "mp4", "mov"]

def add_shared_parser_options(parser: argparse.Namespace) -> None:
    path_args = parser.add_argument_group("paths")
    path_args.add_argument("-i", "--input", default="data/0001_pink_ball_vertical_throw.mp4", type=str)
    path_args.add_argument("-o", "--output-dir", required=False, type=str)

def add_video_parser_args(parser: argparse.Namespace) ->None:
    video_args = parser.add_argument_group("input video")
    video_args.add_argument("-t", "--trim", nargs="+", type=float, help="Trim in seconds like -t 4.8 5.3 or -t 0.5")

def add_visualizer_parser_args(parser: argparse.Namespace) ->None:
    viz_args = parser.add_argument_group("output visualization option")
    viz_args.add_argument("-v", "--visualize", action="store_true")
    viz_args.add_argument("-fps", "--framerate", type=int, default=None, help="visualization fps, 10 if a Gif")

def get_trim(args) -> Tuple[float, float]:
    trim = args.trim
    if trim:
        if len(trim)==1:
            trim = (None,  trim[0])
        else:
            assert len(trim)==2, "trim shall have one or two elements"
        print(f"Video trimming: {trim}")
    return trim