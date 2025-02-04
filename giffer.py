from PIL import Image, ImageDraw, ImageFont
import os


def add_text_overlay(image, text, position=(10, 10), font_size=30):
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    draw.text(position, text, font=font, fill="white", stroke_fill="black", stroke_width=2)

    return image


def create_gif(image_folder, output_gif, duration=200, loop=0):
    images = []

    image_files = sorted(
        [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))],
        key=lambda x: int(''.join(filter(str.isdigit, x)))
    )

    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        img = Image.open(img_path).convert("RGB")

        epoch_number = ''.join(filter(str.isdigit, img_file))
        text = f"Epoch {epoch_number}" if epoch_number else "Unknown Epoch"

        img = add_text_overlay(img, text)

        images.append(img)

    if images:
        images[0].save(output_gif, save_all=True, append_images=images[1:], duration=duration, loop=loop)
        print(f"GIF saved at {output_gif}")
    else:
        print("No images found to create a GIF.")


if __name__ == "__main__":
    image_folder = "gif_directory"
    output_gif = "generated_animation.gif"

    create_gif(image_folder, output_gif)
