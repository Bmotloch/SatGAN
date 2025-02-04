import csv
import os
import time
from PIL import Image
import torchvision.transforms as transforms
import torch
from gan_skip_big import Generator
from sift import stitch_images_sift


def extract_epoch_step(filename):
    parts = filename.split('_')
    epoch = int(parts[2])
    step = int(parts[3].split('.')[0]) if len(parts) > 3 else float('inf')
    return epoch, step


def find_input_images(target_filename, input_dir):
    base_name = target_filename.split('.')[0].split('_')
    row_start = int(base_name[1])
    col_start = int(base_name[2])
    row_end = int(base_name[3])
    col_end = int(base_name[4])
    input1_path = os.path.join(input_dir, f'subtile_{row_start}_{col_start}_left.png')
    input2_path = os.path.join(input_dir, f'subtile_{row_end}_{col_end}_right.png')
    return input1_path, input2_path

def find_input_images_sift(target_filename, input_dir):
    base_name = target_filename.split('.')[0].split('_')
    row_start = int(base_name[1])
    col_start = int(base_name[2])
    row_end = int(base_name[3])
    col_end = int(base_name[4])
    input1_path = os.path.join(input_dir, f'subtile_{row_start}_{col_start}.png')
    input2_path = os.path.join(input_dir, f'subtile_{row_end}_{col_end}.png')
    return input1_path, input2_path


def test_single_input(generator, inputs, device='cuda'):
    inputs = inputs.unsqueeze(0).to(device)

    start_time = time.time()
    with torch.no_grad():
        generated_image = generator(inputs)
    generated_image = ((generated_image + 1) / 2) * 255
    generation_time = time.time() - start_time
    return generated_image, generation_time


def process_image_stitching(input1_path, input2_path):
    import cv2
    img1 = cv2.imread(input1_path)
    img2 = cv2.imread(input2_path)

    start_time = time.time()
    stitched_img = stitch_images_sift(img1, img2, 'row')
    stitching_time = time.time() - start_time

    return stitched_img, stitching_time


if __name__ == "__main__":
    input_dir = 'Philadelphia subtiles padded 896_512'
    input_sift = 'Philadelphia subtiles 512'
    target_dir = 'stitched row 512 test'
    mask_path = 'horizontal_mask.png'
    eval_csv_filename = 'timing_test.csv'
    save_dir = "model_checkpoints_big"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    mask = Image.open(mask_path)
    mask = transform(mask)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)

    os.makedirs("generated_outputs", exist_ok=True)

    generator_files = [f for f in os.listdir(save_dir) if f.startswith('generator_epoch_') and f.endswith('.pth')]
    generator_files.sort(key=lambda x: extract_epoch_step(x))

    with open(eval_csv_filename, mode='a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Epoch', 'Step', 'Target Image', 'Generation Time (s)', 'Stitching Time (s)'])

        for gen_file in generator_files:
            epoch, step = extract_epoch_step(gen_file)

            if epoch != 172:
                continue

            latest_gen = os.path.join(save_dir, gen_file)
            print(f"Loaded {latest_gen}")
            generator.load_state_dict(torch.load(latest_gen, weights_only=True))
            generator.eval()

            for target_filename in os.listdir(target_dir):
                if not target_filename.endswith('.tiff'):
                    continue

                input1_path, input2_path = find_input_images(target_filename, input_dir)
                input1sift_path, input2sift_path = find_input_images_sift(target_filename, input_sift)
                if not os.path.exists(input1_path) or not os.path.exists(input2_path) or not os.path.exists(
                        input1sift_path) or not os.path.exists(input2sift_path):
                    print(f"Input images not found for target {target_filename}")
                    continue

                input_img1 = Image.open(input1_path)
                input_img2 = Image.open(input2_path)

                input_img1 = transform(input_img1)
                input_img2 = transform(input_img2)

                inputs = torch.cat([input_img1, input_img2, mask], dim=0)

                generated_image, generation_time = test_single_input(generator, inputs, device=device)

                stitched_img, stitching_time = process_image_stitching(input1sift_path, input2sift_path)

                csv_writer.writerow([epoch, step, target_filename, generation_time, stitching_time])

                print(f"Epoch: {epoch}, Step: {step}, Target: {target_filename}, "
                      f"Generation Time: {generation_time:.4f}s, Stitching Time: {stitching_time:.4f}s")

    print(f"Evaluation results saved to {eval_csv_filename}")
