from PIL import Image

def load_image(image_path, max_size):
    # Load image and resize it to max_size;
    # returns the resized PIL image, its original and new size
    image = Image.open(image_path).convert('RGB')

    orig_size = image.size
    if orig_size[0] > max_size or orig_size[1] > max_size:
        image = image.resize((max_size, max_size), Image.LANCZOS)

    return image, orig_size, image.size

def save_image(image, image_path, orig_size, show_image=True):
    # Save PIL image and show it
    new_size = image.size
    if new_size[0] < orig_size[0] or new_size[1] < orig_size[1]:
        image = image.resize(orig_size, Image.BICUBIC)

    image.save(image_path)
    if show_image:
        image.show()
