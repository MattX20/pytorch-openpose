
import argparse
import sys
from pathlib import Path
from shared import add_video_parser_args, add_visualizer_parser_args, get_trim, VIDEO_EXT
from video_processing import process_video_frames
from visualize_results import encode_debug_figures

from batch_processing import Batch
from unified_path import append_stem
from main import get_model
from moviepy.editor import VideoFileClip
import logging

def parse_command_line(batch: Batch) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Batch video processing - OpenPose',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_video_parser_args(parser)
    add_visualizer_parser_args(parser)
    parser.add_argument("-vsuf", "--visualization-suffix", default="gif", choices=["gif"] + VIDEO_EXT)
    parser.add_argument("-mp", "--multi-processing", action="store_true", help="Enable multiprocessing - Warning with GPU - use -j2")    
    parser.add_argument("-skip", "--skip-existing", action="store_true", help="skip existing processed folders")
    return batch.parse_args(parser)


def parallel_process(input: Path, output: Path, args: argparse.Namespace, model=None):
    if model is None:
        model = get_model() # Load model for each thread , be careful!
    print(input, output.parent)
    trim = get_trim(args)
    if output.exists() and args.skip_existing:
            logging.warning(f"Results already exist - skip processing  {output}")
    else:
        logging.warning(f"Reprocessing found results - use --skip-existing to skip processing  {output}")
        output.mkdir(parents=True, exist_ok=True)
        process_video_frames(input, output, trim=trim)
    
    if not args.framerate:
        fps = VideoFileClip(str(input)).fps
        print(f"Auto framerate deduced: {fps}")
    else:
        fps = args.framerate
    if args.visualize:
        encode_debug_figures(
            output,
            append_stem(output, f"__pose_estimation_{fps:d}fps").with_suffix("."+args.visualization_suffix),
            fps=fps
        )

def main(argv):
    # Instantiate batch
    batch = Batch(argv)
    batch.set_io_description(input_help='input video files', output_help='output directory')
    
    
    # Parse arguments
    args = parse_command_line(batch)
    multiprocessing = args.multi_processing
    # Disable mp - Highly recommended!
    if multiprocessing:
        model = None
    else:
        # Disable multiprocessing -> single 
        batch.set_multiprocessing_enabled(False)
        model = get_model() # Create the model only one
    batch.run(parallel_process, model)



if __name__ == "__main__":
    main(sys.argv[1:])