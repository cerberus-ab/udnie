# Udnie stylish
Applies artistic style transfer to images using
- pre-trained VGG-19 model and optimization-based NST algorithm (L-BFGS or Adam)

### Usage
You need Python 3.9 or later, then requirements:
```sh
pip install -r requirements.txt
```

To run:
```sh
python src/nst.py --help
python src/nst.py --input=<path> [--style=<style>]
```

### Examples
Input image:  
<img src="data/input/example.jpg" alt="input" width="256">

Style images:  
<img src="data/ref/udnie.jpg" alt="style-udnie" width="256">
<img src="data/ref/X.jpg" alt="style-X" width="256">
<img src="data/ref/starry_night.jpg" alt="style-starry_night" width="256">

Output images:  
<img src="data/output/example-udnie.jpg" alt="output-udnie" width="256">
<img src="data/output/example-X.jpg" alt="output-X" width="256">
<img src="data/output/example-starry_night.jpg" alt="output-starry_night" width="256">
