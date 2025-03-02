import argparse
import os
from dataclasses import dataclass

# Constants
_style_map = {
    "udnie": "data/ref/udnie-512-a.jpg",
    "sunflowers": "data/ref/sunflowers-512.jpg",
    "17a": "data/ref/17a-512.jpg",
    "black-square": "data/ref/black-square-512.jpg",
}
_output_dir = "data/output"

def _get_output_paths(input_path, style_name, steps):
    # Creates output paths for styled and post-processed files
    input_name, input_ext = os.path.splitext(os.path.basename(input_path))
    input_pref_name = f"{input_name}-{style_name}-{steps}"
    output_path = f"{_output_dir}/{input_pref_name}{input_ext}"
    output_post_path = f"{_output_dir}/{input_pref_name}-post{input_ext}"

    return output_path, output_post_path

@dataclass
class Params:
    style_name: str
    style_path: str
    input_path: str
    output_path: str
    output_post_path: str
    post_process: bool
    steps: int

    @staticmethod
    def of_args():
        parser = argparse.ArgumentParser(
            prog="Udnie",
            description="Applies artistic style transfer to images using a pre-trained VGG-19 model."
        )
        parser.add_argument("--input", type=str, required=True,
                            help="Path to the input image")
        parser.add_argument("--style", type=str, choices=_style_map.keys(), default="udnie",
                            help=f"Style reference")
        parser.add_argument("--post", action="store_true",
                            help="Apply post-processing to the styled image")
        parser.add_argument("-s", "--steps", type=int, default=150,
                            help="Number of optimization steps")

        args = parser.parse_args()
        output_path, output_post_path = _get_output_paths(args.input, args.style, args.steps)

        return Params(
            args.style,
            _style_map[args.style],
            args.input,
            output_path,
            output_post_path,
            args.post,
            args.steps)
