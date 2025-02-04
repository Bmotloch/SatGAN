import re
from PIL import Image
import os
import random
import shutil


def extract_row_col_filename(filename):
    base_name = os.path.basename(filename)
    match = re.search(r"subtile_(\d+)_(\d+)", base_name)
    if match:
        return int(match.group(1)), int(match.group(2))
    else:
        print(f"Filename does not match pattern: {filename}")
        return None, None

def pad_images(input_dir, horizontal_output_dir, vertical_output_dir):
    os.makedirs(horizontal_output_dir, exist_ok=True)
    os.makedirs(vertical_output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            input_path = os.path.join(input_dir, filename)
            try:
                image = Image.open(input_path)

                if image.size != (512, 512):
                    print(f"Skipping non-square image: {filename}")
                    continue

                row, col = extract_row_col_filename(filename)
                if row is None or col is None:
                    continue

                # # Add horizontal padding
                # horizontal_image = Image.new("RGB", (896, 512), (0, 0, 0))
                # horizontal_image.paste(image, (0, 0))
                # horizontal_output_path = os.path.join(horizontal_output_dir, f"subtile_{row}_{col}_left.png")
                # horizontal_image.save(horizontal_output_path)

                # horizontal_image = Image.new("RGB", (896, 512), (0, 0, 0))
                # horizontal_image.paste(image, (384, 0))
                # horizontal_output_path = os.path.join(horizontal_output_dir, f"subtile_{row}_{col}_right.png")
                # horizontal_image.save(horizontal_output_path)

                # Add vertical padding
                vertical_image = Image.new("RGB", (512, 896), (0, 0, 0))
                vertical_image.paste(image, (0, 0))
                vertical_output_path = os.path.join(vertical_output_dir, f"subtile_{row}_{col}_up.png")
                vertical_image.save(vertical_output_path)

                vertical_image = Image.new("RGB", (512, 896), (0, 0, 0))
                vertical_image.paste(image, (0, 384))
                vertical_output_path = os.path.join(vertical_output_dir, f"subtile_{row}_{col}_down.png")
                vertical_image.save(vertical_output_path)

                print(f"Processed: {filename}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")


def create_mask(width, height, output_path):
    mask = Image.new("L", (width, height), 0)
    draw = mask.load()

    if width > height:
        for y in range(height):
            for x in range(384, 384 + 128):
                draw[x, y] = 255
    else:
        for x in range(width):
            for y in range(384, 384 + 128):
                draw[x, y] = 255

    mask.save(output_path)
    print(f"Mask saved at: {output_path}")

def move_images(src_dir, dest_dir, percentage=10, seed=42):
    os.makedirs(dest_dir, exist_ok=True)

    image_extensions = ('.png', '.jpg', '.jpeg', '.tiff')
    images = [f for f in os.listdir(src_dir) if f.lower().endswith(image_extensions)]

    num_to_move = max(1, int(len(images) * (percentage / 100)))
    random.seed(seed)
    random.shuffle(images)
    files_to_move = images[:num_to_move]
    for file_name in files_to_move:
        src_path = os.path.join(src_dir, file_name)
        dest_path = os.path.join(dest_dir, file_name)
        if not os.path.exists(dest_path):
            shutil.move(src_path, dest_path)
        else:
            print(f"File {file_name} already exists in {dest_dir}, skipping.")

    print(f"Moved {len(files_to_move)} files from {src_dir} to {dest_dir}.")

# move_images('stitched row 512', 'stitched row 512 test', percentage=10, seed=42)


# input_directory = "Philadelphia subtiles 512"
# horizontal_output_directory = "Philadelphia subtiles padded 896_512"
# vertical_output_directory = "Philadelphia subtiles padded 512_896"
# mask_horizontal_path = "horizontal_mask.png"
# mask_vertical_path = "vertical_mask.png"

# pad_images(input_directory, horizontal_output_directory, vertical_output_directory)
# create_mask(512, 896, mask_horizontal_path)
# create_mask(896, 512, mask_vertical_path)

