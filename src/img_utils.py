from PIL import Image

def load_image(image_path, max_size=None):
    # Load image and resize it to max_size;
    # returns the resized PIL image and its original size
    image = Image.open(image_path).convert('RGB')

    w, h = image.size
    if max_size is not None and max(w, h) > max_size:
        scale_factor = max_size / max(w, h)
        new_size = (int(w * scale_factor), int(h * scale_factor))
        image = image.resize(new_size, Image.LANCZOS)

    return image, (w, h)

def save_image(image, image_path, orig_size, show_image=True):
    # Save PIL image and show it
    new_size = image.size
    if new_size[0] < orig_size[0] or new_size[1] < orig_size[1]:
        image = image.resize(orig_size, Image.BICUBIC)

    image.save(image_path)
    if show_image:
        image.show()
