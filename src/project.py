import os
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from PIL import Image
from tqdm import tqdm

from generator import Generator
from discriminator import Discriminator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def train_cyclegan(style_dir, epochs=100, batch_size=1, lr=0.0002):
    # gotta resize our data....
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load datasets
    dataset_A = datasets.ImageFolder(style_dir, transform=transform)
    dataset_B = datasets.ImageFolder("images/input", transform=transform)
    
    loader_A = DataLoader(dataset_A, batch_size=batch_size, shuffle=True)
    loader_B = DataLoader(dataset_B, batch_size=batch_size, shuffle=True)

    # initialize models
    G = Generator().to(device)  # A to B
    F = Generator().to(device)  # B to A

    D_A = Discriminator().to(device)  # Real or fake for domain A
    D_B = Discriminator().to(device)  # Real or fake for domain B

    # set up optimizers
    opt_G = optim.Adam(list(G.parameters()) + list(F.parameters()), lr=lr, betas=(0.5, 0.999))
    opt_D_A = optim.Adam(D_A.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D_B = optim.Adam(D_B.parameters(), lr=lr, betas=(0.5, 0.999))

    # GAN loss
    criterion_gan = nn.MSELoss()

    # cycle consistency loss 
    criterion_cycle = nn.L1Loss()

    # identity mapping loss
    criterion_identity = nn.L1Loss() 

    # train
    for epoch in range(epochs):
        print(f"Epoch [{epoch + 1}/{epochs}]")
        for (real_A, _), (real_B, _) in tqdm(zip(loader_A, loader_B), total=min(len(loader_A), len(loader_B))):
            real_A, real_B = real_A.to(device), real_B.to(device)

            # train generators G and F
            opt_G.zero_grad()
            fake_B = G(real_A)
            fake_A = F(real_B)
            cycle_A = F(fake_B)
            cycle_B = G(fake_A)
            identity_A = F(real_A)
            identity_B = G(real_B)

            loss_gan_G = criterion_gan(D_B(fake_B), torch.ones_like(D_B(fake_B)))  # G tries to fool D_B
            loss_gan_F = criterion_gan(D_A(fake_A), torch.ones_like(D_A(fake_A)))  # F tries to fool D_A
            loss_cycle = criterion_cycle(cycle_A, real_A) + criterion_cycle(cycle_B, real_B)
            loss_identity = criterion_identity(identity_A, real_A) + criterion_identity(identity_B, real_B)

            loss_G = loss_gan_G + loss_gan_F + 10 * loss_cycle + 5 * loss_identity
            loss_G.backward()
            opt_G.step()

            # train discriminators D_A and D_B
            opt_D_A.zero_grad()
            loss_D_A = (criterion_gan(D_A(real_A), torch.ones_like(D_A(real_A))) +
                        criterion_gan(D_A(fake_A.detach()), torch.zeros_like(D_A(fake_A)))) * 0.5
            loss_D_A.backward()
            opt_D_A.step()

            opt_D_B.zero_grad()
            loss_D_B = (criterion_gan(D_B(real_B), torch.ones_like(D_B(real_B))) +
                        criterion_gan(D_B(fake_B.detach()), torch.zeros_like(D_B(fake_B)))) * 0.5
            loss_D_B.backward()
            opt_D_B.step()

        print(f"Epoch {epoch + 1}/{epochs} completed!")

        # save model checkpoints
        torch.save(G.state_dict(), f"checkpoints/G_epoch_{epoch + 1}.pth")
        torch.save(F.state_dict(), f"checkpoints/F_epoch_{epoch + 1}.pth")
        torch.save(D_A.state_dict(), f"checkpoints/D_A_epoch_{epoch + 1}.pth")
        torch.save(D_B.state_dict(), f"checkpoints/D_B_epoch_{epoch + 1}.pth")

    print("Training completed!")

def main(args):
    if args.train:
        train_cyclegan(args.style_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    else:
        # Inference for style transfer
        style_dir = args.style_dir
        input_image_path = args.input_image
        output_image_path = args.output_image

        if not os.path.exists(style_dir):
            raise FileNotFoundError(f"Style directory not found: {style_dir}")
        if not os.path.exists(input_image_path):
            raise FileNotFoundError(f"Input image not found: {input_image_path}")
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        input_image = load_image(input_image_path, transform).to(device)
        G = Generator().to(device)
        G.load_state_dict(torch.load('checkpoints/G_epoch_100.pth', map_location=device, weights_only=False))
        G.eval()

        with torch.no_grad():
            styled_image = G(input_image)

        styled_image = styled_image.cpu().squeeze(0)
        styled_image = transforms.ToPILImage()(styled_image * 0.5 + 0.5)
        styled_image.save(output_image_path)
        print(f"Styled image saved at: {output_image_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CycleGAN Style Transfer and Training")
    parser.add_argument('--style_dir', type=str, default='images/art/vangogh/', help="Path to the style images directory.")
    parser.add_argument('--input_image', type=str, default='images/input/input.jpg', help="Path to the input image to style transfer.")
    parser.add_argument('--output_image', type=str, default='images/output/styled_image.jpg', help="Path to save the styled output image.")
    parser.add_argument('--train', action='store_true', help="Flag to train the CycleGAN instead of performing inference.")
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs for training.")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for training.")
    parser.add_argument('--lr', type=float, default=0.0002, help="Learning rate for training.")
    args = parser.parse_args()
    main(args)
