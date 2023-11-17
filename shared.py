import argparse
VIDEO_EXT = ["mp4", "avi", "mp4", "mov"]

def add_shared_parser_options(parser: argparse.Namespace) -> None:
    path_args = parser.add_argument_group("paths")
    path_args.add_argument("-i", "--input", default="data/0001_pink_ball_vertical_throw.mp4", type=str)
    path_args.add_argument("-o", "--output-dir", required=False, type=str)