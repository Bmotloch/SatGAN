import csv
import os
from PIL import Image
import torchvision.transforms as transforms
import torch
from gan_skip_big import Generator
import torch.nn.functional as functional


def calculate_mse(img1, img2):
    return functional.mse_loss(img1, img2).item()


def calculate_psnr(mse, max_val=255.0):
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(torch.tensor((max_val ** 2) / mse)).item()


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


def test_single_input(generator, inputs, target, device='cuda'):
    generator.to(device)
    generator.eval()

    inputs = inputs.unsqueeze(0).to(device)
    target = target.unsqueeze(0).to(device)

    with torch.no_grad():
        generated_image = generator(inputs)

    generated_image = ((generated_image + 1) / 2) * 255
    target = ((target + 1) / 2) * 255

    mse = calculate_mse(generated_image, target)
    psnr = calculate_psnr(mse)

    return mse, psnr


if __name__ == "__main__":
    input_dir = 'Philadelphia subtiles padded 896_512'
    target_dir = 'stitched row 512 test'
    mask_path = 'horizontal_mask.png'
    eval_csv_filename = 'test_eval.csv'
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
        csv_writer.writerow(['Epoch', 'Step', 'Target Image', 'MSE', 'PSNR'])

        for gen_file in generator_files:
            latest_gen = os.path.join(save_dir, gen_file)
            epoch, step = extract_epoch_step(gen_file)

            print(f"Loaded {latest_gen}")
            generator.load_state_dict(torch.load(latest_gen, weights_only=True))
            generator.eval()

            for target_filename in os.listdir(target_dir):
                if not target_filename.endswith('.tiff'):
                    continue

                input1_path, input2_path = find_input_images(target_filename, input_dir)
                if not os.path.exists(input1_path) or not os.path.exists(input2_path):
                    print(f"Input images not found for target {target_filename}")
                    continue

                input_img1 = Image.open(input1_path)
                input_img2 = Image.open(input2_path)
                target_img = Image.open(os.path.join(target_dir, target_filename))

                input_img1 = transform(input_img1)
                input_img2 = transform(input_img2)
                target_img = transform(target_img)

                inputs = torch.cat([input_img1, input_img2, mask], dim=0)

                mse, psnr = test_single_input(generator, inputs, target_img, device=device)

                csv_writer.writerow([epoch, step, target_filename, mse, psnr])
                print(f"Epoch: {epoch}, Step: {step}, Target: {target_filename}, MSE: {mse}, PSNR: {psnr}")

    print(f"Evaluation results saved to {eval_csv_filename}")
