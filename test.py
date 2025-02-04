from PIL import Image, ImageOps
import torchvision.transforms as transforms
import torch
from torchvision.transforms.functional import to_pil_image
from gan_skip_big import Generator
import os


def extract_epoch_step(filename):
    parts = filename.split('_')
    epoch = int(parts[2])
    step = int(parts[3].split('.')[0]) if len(parts) > 3 else float('inf')
    return epoch, step


def test_single_input(generator, inputs, device='cuda', output_path='output.png'):
    generator.to(device)
    generator.eval()

    inputs = inputs.unsqueeze(0).to(device)

    with torch.no_grad():
        generated_image = generator(inputs)

    generated_image = (generated_image + 1) / 2

    image = to_pil_image(generated_image[0].cpu())
    image.save(output_path)

    print(f"Generated image saved to {output_path}")


if __name__ == "__main__":
    mask_path = 'horizontal_mask.png'
    #input_path1 = 'Philadelphia subtiles padded 896_512/subtile_0_9_left.png'
    #input_path2 = 'Philadelphia subtiles padded 896_512/subtile_0_10_right.png'
    input_path1 = 'Philadelphia subtiles padded 896_512/subtile_0_0_left.png'
    input_path2 = 'Philadelphia subtiles padded 896_512/subtile_0_1_right.png'
    # input_path1 = 'Philadelphia subtiles padded 896_512/subtile_5_95_left.png'
    # input_path2 = 'Philadelphia subtiles padded 896_512/subtile_5_96_right.png'
    # input_path1 = 'Philadelphia subtiles padded 896_512/subtile_56_58_left.png'
    # input_path2 = 'Philadelphia subtiles padded 896_512/subtile_56_59_right.png'

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    mask_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    input_img1 = Image.open(input_path1)
    input_img2 = Image.open(input_path2)
    mask = Image.open(mask_path)

    input_img1 = transform(input_img1)
    input_img2 = transform(input_img2)
    mask = mask_transform(mask)

    inputs = torch.cat([input_img1, input_img2, mask], dim=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)
    save_dir = "model_checkpoints_big"
    os.makedirs(save_dir, exist_ok=True)

    generator_files = [f for f in os.listdir(save_dir) if f.startswith('generator_epoch_') and f.endswith('.pth')]

    last_epoch = 0
    if generator_files:
        generator_files.sort(key=lambda x: extract_epoch_step(x))
        latest_gen = os.path.join(save_dir, generator_files[-1])
        last_epoch, _ = extract_epoch_step(generator_files[-1])

        print(f"Loaded {latest_gen}")
        # generator.load_state_dict(torch.load(latest_gen, weights_only=True))
        generator.load_state_dict(torch.load("model_checkpoints_big/generator_epoch_172_562.pth", weights_only=True))
    generator.eval()

    test_single_input(generator, inputs, device='cuda', output_path='test_output_big.png')
