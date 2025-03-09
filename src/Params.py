import argparse
import os
from dataclasses import dataclass

# Constants
_style_map = {
    "udnie": "data/ref/udnie.jpg",
    "sunflowers": "data/ref/sunflowers.jpg",
    "starry_night": "data/ref/starry_night.jpg",
    "X": "data/ref/X.jpg",
    "17a": "data/ref/17a.jpg",
}
_output_dir = "data/output"

def _get_output_path(input_path, style_name, optim, steps):
    # Creates output path for styled file
    input_name, input_ext = os.path.splitext(os.path.basename(input_path))
    input_pref_name = f"{input_name}-{style_name}-{optim}-{steps}"
    output_path = f"{_output_dir}/{input_pref_name}{input_ext}"

    return output_path

@dataclass
class Params:
    style_name: str
    style_path: str
    input_path: str
    output_path: str
    device_type: str
    optim: str
    steps: int
    size: int
    show: bool

    @staticmethod
    def of_args():
        parser = argparse.ArgumentParser(
            prog="Udnie",
            description="Applies artistic style transfer to images"
        )
        parser.add_argument("--input", type=str, required=True,
                            help="Path to the input image")
        parser.add_argument("--style", type=str, choices=_style_map.keys(), default="udnie",
                            help="Style reference, By default: udnie")
        parser.add_argument("--device", type=str, choices=("cpu", "cuda", "mps", "tpu"), default=None,
                            help="Device type to run the model on, By default: the best available")
        parser.add_argument("--optim", type=str, choices=("lbfgs", "adam"), default="lbfgs",
                            help="Optimization algorithm, By default: lbfgs")
        parser.add_argument("--steps", type=int, default=300,
                            help="Number of optimization steps, By default: 300")
        parser.add_argument("--size", type=int, default=512,
                            help="Size of the images in the optimization, By default: 512")
        parser.add_argument("--show", action="store_true",
                            help="Show the styled image")

        args = parser.parse_args()
        output_path = _get_output_path(args.input, args.style, args.optim, args.steps)

        return Params(
            args.style,
            _style_map[args.style],
            args.input,
            output_path,
            args.device,
            args.optim,
            args.steps,
            args.size,
            args.show)
