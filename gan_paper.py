import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
from PIL import Image
import csv

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1 or classname.find('InstanceNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(7, 32, kernel_size=8, stride=2, padding=3),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=8, stride=2, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(64) for _ in range(4)]
        )

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=8, stride=2, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(64 + 32, 32, kernel_size=8, stride=2, padding=3),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)

        res_output = enc2
        for block in self.residual_blocks:
            res_output = block(res_output)

        dec1 = self.decoder1(res_output)
        dec2 = self.decoder2(torch.cat([dec1, enc1], dim=1))
        out = self.final(dec2)

        return out


class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.model(x)


class StitchDataset(Dataset):
    def __init__(self, input_dir, stitched_dir, transform=None, mode='row'):
        self.input_dir = input_dir
        self.stitched_dir = stitched_dir
        self.transform = transform
        self.mode = mode

        self.input_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
        self.stitched_files = [f for f in os.listdir(stitched_dir) if f.endswith('.tiff')]

        self.mapping = self.create_mapping()

    def create_mapping(self):
        mapping = []
        for stitched_file in self.stitched_files:
            base_name = stitched_file.split('.')[0].split('_')
            row_start = int(base_name[1])
            col_start = int(base_name[2])
            row_end = int(base_name[3])
            col_end = int(base_name[4])

            if mode == 'row':
                input_file1 = f"subtile_{row_start}_{col_start}_left.png"
                input_file2 = f"subtile_{row_end}_{col_end}_right.png"
            else:
                input_file1 = f"subtile_{row_start}_{col_start}_up.png"
                input_file2 = f"subtile_{row_end}_{col_end}_down.png"

            if input_file1 in self.input_files and input_file2 in self.input_files:
                mapping.append((input_file1, input_file2, stitched_file))
        return mapping

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        input_file1, input_file2, stitched_file = self.mapping[idx]

        input_path1 = os.path.join(self.input_dir, input_file1)
        input_path2 = os.path.join(self.input_dir, input_file2)
        stitched_path = os.path.join(self.stitched_dir, stitched_file)

        input_img1 = Image.open(input_path1)
        input_img2 = Image.open(input_path2)

        if self.mode == 'row':
            input_mask = Image.open('horizontal_mask.png')
        else:
            input_mask = Image.open('vertical_mask.png')

        stitched_img = Image.open(stitched_path)

        if self.transform:
            input_img1 = self.transform(input_img1)
            input_img2 = self.transform(input_img2)
            mask_img = self.transform(input_mask)
            stitched_img = self.transform(stitched_img)

        inputs = torch.cat([input_img1, input_img2, mask_img], dim=0)
        return inputs, stitched_img


def train_gan(generator, discriminator, dataloader, num_epochs=10, device='cuda', last_epoch=0, save_dir="checkpoints",
              log_file="training_log_paper.csv"):
    accumulation_steps = 2
    criterion = nn.BCEWithLogitsLoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0005, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0005, betas=(0.5, 0.999))
    scaler = torch.amp.GradScaler(device=device)

    generator.to(device)
    discriminator.to(device)

    if not os.path.exists(log_file):
        with open(log_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Step", "Loss_D", "Loss_Real", "Loss_Fake", "Loss_G"])

    for epoch in range(last_epoch, num_epochs + last_epoch):
        for i, (inputs, real_stitched) in enumerate(dataloader):
            inputs, real_stitched = inputs.to(device), real_stitched.to(device)

            cumulative_loss_real = 0.0
            cumulative_loss_fake = 0.0
            cumulative_loss_D = 0.0
            cumulative_loss_G = 0.0

            optimizer_D.zero_grad(set_to_none=True)
            for j in range(accumulation_steps):
                micro_inputs = inputs[j::accumulation_steps]
                micro_real_stitched = real_stitched[j::accumulation_steps]

                with torch.amp.autocast(device_type='cuda'):
                    fake_stitched = generator(micro_inputs)
                    real_output = discriminator(micro_real_stitched)
                    fake_output = discriminator(fake_stitched.detach())

                    real_labels = torch.ones_like(real_output)
                    fake_labels = torch.zeros_like(fake_output)
                    loss_real = criterion(real_output, real_labels)
                    loss_fake = criterion(fake_output, fake_labels)
                    loss_D = (loss_real + loss_fake) / 2

                scaler.scale(loss_D / accumulation_steps).backward()
                cumulative_loss_real += loss_real.item()
                cumulative_loss_fake += loss_fake.item()
                cumulative_loss_D += loss_D.item()

            scaler.step(optimizer_D)
            scaler.update()

            optimizer_G.zero_grad(set_to_none=True)
            for j in range(accumulation_steps):
                micro_inputs = inputs[j::accumulation_steps]

                with torch.amp.autocast(device_type='cuda'):
                    fake_stitched = generator(micro_inputs)
                    fake_output = discriminator(fake_stitched)
                    loss_G = criterion(fake_output, torch.ones_like(fake_output))

                scaler.scale(loss_G / accumulation_steps).backward()
                cumulative_loss_G += loss_G.item()

            scaler.step(optimizer_G)
            scaler.update()

            avg_loss_real = cumulative_loss_real / accumulation_steps
            avg_loss_fake = cumulative_loss_fake / accumulation_steps
            avg_loss_D = cumulative_loss_D / accumulation_steps
            avg_loss_G = cumulative_loss_G / accumulation_steps

            print(f"Epoch [{epoch + 1}/{num_epochs + last_epoch}], Step [{i + 1}/{len(dataloader)}], "
                  f"Loss D: {avg_loss_D}, Loss Real: {avg_loss_real}, Loss Fake: {avg_loss_fake}, Loss G: {avg_loss_G}")

            with open(log_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch + 1, i + 1, avg_loss_D, avg_loss_real, avg_loss_fake, avg_loss_G])

            if (i + 1) % 281 == 0:
                torch.save(generator.state_dict(), os.path.join(save_dir, f"generator_epoch_{epoch + 1}_{i + 1}.pth"))
                torch.save(discriminator.state_dict(),
                           os.path.join(save_dir, f"discriminator_epoch_{epoch + 1}_{i + 1}.pth"))


if __name__ == "__main__":
    input_dir = 'Philadelphia subtiles padded 896_512'
    stitched_dir = 'stitched row 512 train'
    mode = 'row'

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = StitchDataset(input_dir, stitched_dir, transform=transform, mode=mode)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    save_dir = "model_checkpoints_skip1"
    os.makedirs(save_dir, exist_ok=True)

    generator_files = [f for f in os.listdir(save_dir) if f.startswith('generator_epoch_') and f.endswith('.pth')]
    discriminator_files = [f for f in os.listdir(save_dir) if
                           f.startswith('discriminator_epoch_') and f.endswith('.pth')]

    last_epoch = 0
    if generator_files and discriminator_files:
        def extract_epoch_step(filename):
            parts = filename.split('_')
            epoch = int(parts[2])
            step = int(parts[3].split('.')[0]) if len(parts) > 3 else float('inf')
            return epoch, step


        generator_files.sort(key=lambda x: extract_epoch_step(x))
        discriminator_files.sort(key=lambda x: extract_epoch_step(x))

        latest_gen = os.path.join(save_dir, generator_files[-1])
        latest_dis = os.path.join(save_dir, discriminator_files[-1])

        last_epoch, _ = extract_epoch_step(generator_files[-1])
        print(f"Resuming from epoch {last_epoch}.")

        generator.load_state_dict(torch.load(latest_gen, weights_only=True))
        discriminator.load_state_dict(torch.load(latest_dis, weights_only=True))

    train_gan(generator, discriminator, dataloader, num_epochs=5, device=device, last_epoch=last_epoch,
              save_dir=save_dir, log_file="training_log_paper.csv")
