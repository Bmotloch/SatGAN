import os
from PIL import Image
import torchvision.transforms as transforms
import torch
from gan_skip_big import Generator

def extract_epoch_step(filename):
    parts = filename.split('_')
    epoch = int(parts[2])
    step = int(parts[3].split('.')[0]) if len(parts) > 3 else float('inf')
    return epoch, step

def test_single_input(generator, inputs, device='cuda'):
    generator.to(device)
    generator.eval()

    inputs = inputs.unsqueeze(0).to(device)

    with torch.no_grad():
        generated_image = generator(inputs)

    generated_image = ((generated_image + 1) / 2)

    generated_image_pil = transforms.ToPILImage()(generated_image.squeeze(0).cpu())

    return generated_image_pil

if __name__ == "__main__":
    input_path1 = 'Philadelphia subtiles padded 896_512/subtile_0_9_left.png'
    input_path2 = 'Philadelphia subtiles padded 896_512/subtile_0_10_right.png'
    mask_path = 'horizontal_mask.png'
    save_dir = "model_checkpoints_big"
    output_dir = 'gif_directory'

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    mask = Image.open(mask_path)
    mask = transform(mask)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)

    os.makedirs(output_dir, exist_ok=True)

    generator_files = [f for f in os.listdir(save_dir) if f.startswith('generator_epoch_') and f.endswith('.pth')]
    generator_files.sort(key=lambda x: extract_epoch_step(x))

    for gen_file in generator_files:
        latest_gen = os.path.join(save_dir, gen_file)
        epoch, step = extract_epoch_step(gen_file)

        print(f"Loaded {latest_gen}")
        generator.load_state_dict(torch.load(latest_gen, map_location=device))
        generator.eval()

        input_img1 = Image.open(input_path1)
        input_img2 = Image.open(input_path2)

        input_img1 = transform(input_img1)
        input_img2 = transform(input_img2)

        inputs = torch.cat([input_img1, input_img2, mask], dim=0)

        generated_image_pil = test_single_input(generator, inputs, device=device)

        output_path = os.path.join(output_dir, f'generated_epoch_{epoch}.png')
        generated_image_pil.save(output_path)
        print(f"Saved generated image for epoch {epoch} at {output_path}")
