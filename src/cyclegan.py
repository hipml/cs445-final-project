import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import random
import itertools

# Generator architecture
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        # pad -> conv -> normalize -> relu
        #     -> pad -> conv -> normalize
        self.block = nn.Sequential(

            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, input_channels=3, num_residuals=9):
        super().__init__()

        CHANNELS = 64 # from original paper
        KERNEL_SIZE = 7 # 7x7 from original paper
        
        # Initial convolution
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, CHANNELS, KERNEL_SIZE),
            nn.InstanceNorm2d(CHANNELS),
            nn.ReLU(inplace=True)
        )
        
        # Downsampling
        self.down_sampling = nn.Sequential(
            nn.Conv2d(CHANNELS, 2*CHANNELS, 3, stride=2, padding=1),
            nn.InstanceNorm2d(2*CHANNELS),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*CHANNELS, 4*CHANNELS, 3, stride=2, padding=1),
            nn.InstanceNorm2d(4*CHANNELS),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        self.residuals = nn.Sequential(
            *[ResidualBlock(4*CHANNELS) for _ in range(num_residuals)]
        )
        
        # Upsampling
        self.up_sampling = nn.Sequential(
            nn.ConvTranspose2d(4*CHANNELS, 2*CHANNELS, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(2*CHANNELS),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(2*CHANNELS, CHANNELS, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(CHANNELS),
            nn.ReLU(inplace=True)
        )
        
        # Final convolution
        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(CHANNELS, input_channels, KERNEL_SIZE),
            # tanh squashes values to [-1, 1] range for valid img output
            nn.Tanh()
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.down_sampling(x)
        x = self.residuals(x)
        x = self.up_sampling(x)
        return self.final(x)

# Discriminator architecture
class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()

        def discriminator_block(in_channels, out_channels, normalize=True):
            layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        CHANNELS = 64
        self.model = nn.Sequential(
            *discriminator_block(input_channels, CHANNELS, normalize=False),
            *discriminator_block(CHANNELS, 2*CHANNELS),
            *discriminator_block(2*CHANNELS, 4*CHANNELS),
            *discriminator_block(4*CHANNELS, 8*CHANNELS),
            nn.Conv2d(8*CHANNELS, 1, 4, padding=1)
        )

    def forward(self, x):
        return self.model(x)

# Custom dataset
class ArtisticDataset(Dataset):
    def __init__(self, root_A, root_B, transform=None):
        self.root_A = root_A
        self.root_B = root_B
        self.transform = transform

        # more effective training samples through augmentation
        # crops, flips, rotations, etc
        # self.transform = transforms.Compose([
        #     transforms.Resize(286),  # Larger size for random cropping
        #     transforms.RandomCrop(256),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomRotation(10),
        #     transforms.ColorJitter(brightness=0.1, contrast=0.1),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # ])
        
        self.files_A = [f for f in os.listdir(root_A) if f.endswith(('.jpg', '.jpeg', '.JPG', '.png'))]
        self.files_B = [f for f in os.listdir(root_B) if f.endswith(('.jpg', '.png'))]

        # print(f"Input images: {len(self.files_A)}")
        # print(f"Style images: {len(self.files_B)}")
        # 
        # for i in range(3):
        #     print(f"\nPair {i}:")
        #     print(f"Input: {self.files_A[i % len(self.files_A)]}")
        #     print(f"Style: {self.files_B[random.randint(0, len(self.files_B)-1)]}")
        
    def __getitem__(self, index):
        # Get random images from both domains
        item_A = Image.open(os.path.join(self.root_A, self.files_A[index % len(self.files_A)]))
        item_B = Image.open(os.path.join(self.root_B, self.files_B[random.randint(0, len(self.files_B)-1)]))
        
        if self.transform:
            item_A = self.transform(item_A)
            item_B = self.transform(item_B)
            
        return {'A': item_A, 'B': item_B}
    
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

# Training class
class CycleGAN:
    def __init__(self, root_A, root_B, device='cuda'):
        self.device = device
        
        # Initialize networks
        self.G_AB = Generator().to(device)
        self.G_BA = Generator().to(device)
        self.D_A = Discriminator().to(device)
        self.D_B = Discriminator().to(device)
        
        # Initialize optimizers
        self.optimizer_G = optim.Adam(
            itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()),
            lr=2e-5, betas=(0.5, 0.999)
        )
        self.optimizer_D = optim.Adam(
            itertools.chain(self.D_A.parameters(), self.D_B.parameters()),
            lr=2e-5, betas=(0.5, 0.999)
        )
        
        # Initialize loss functions
        self.criterion_GAN = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        
        # Initialize dataset
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.dataset = ArtisticDataset(root_A, root_B, transform=transform)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=True,
            num_workers=2
        )
        
    def train_step(self, real_A, real_B):
        # Generate fake samples
        fake_B = self.G_AB(real_A)
        fake_A = self.G_BA(real_B)
        
        # Reconstruct samples
        recon_A = self.G_BA(fake_B)
        recon_B = self.G_AB(fake_A)
        
        # Identity loss
        id_A = self.G_BA(real_A)
        id_B = self.G_AB(real_B)
        identity_loss_A = self.criterion_identity(id_A, real_A)
        identity_loss_B = self.criterion_identity(id_B, real_B)
        
        # GAN loss
        D_A_fake = self.D_A(fake_A)
        D_B_fake = self.D_B(fake_B)
        valid = torch.ones_like(D_A_fake).to(self.device)
        fake = torch.zeros_like(D_A_fake).to(self.device)
        
        # Generator losses
        gan_loss_BA = self.criterion_GAN(D_A_fake, valid)
        gan_loss_AB = self.criterion_GAN(D_B_fake, valid)
        
        # Cycle consistency losses
        cycle_loss_A = self.criterion_cycle(recon_A, real_A)
        cycle_loss_B = self.criterion_cycle(recon_B, real_B)
        
        # total generator loss
        hyper1 = 10.0
        hyper2 = 5.0

        loss_G = (
            gan_loss_BA + gan_loss_AB +
            (cycle_loss_A + cycle_loss_B) * hyper1 +
            (identity_loss_A + identity_loss_B) * hyper2 
        )
        
        # Update generators
        self.optimizer_G.zero_grad()
        loss_G.backward()
        self.optimizer_G.step()
        
        # Discriminator A loss
        loss_D_A = (
            self.criterion_GAN(self.D_A(real_A), valid) +
            self.criterion_GAN(self.D_A(fake_A.detach()), fake)
        ) * 0.5
        
        # Discriminator B loss
        loss_D_B = (
            self.criterion_GAN(self.D_B(real_B), valid) +
            self.criterion_GAN(self.D_B(fake_B.detach()), fake)
        ) * 0.5
        
        # Update discriminators
        self.optimizer_D.zero_grad()
        loss_D_A.backward()
        loss_D_B.backward()
        self.optimizer_D.step()
        
        return {
            'loss_G': loss_G.item(),
            'loss_D': (loss_D_A + loss_D_B).item()
        }
    
    def train(self, num_epochs):
        """ Train the model. Save results every 5 epochs, back up on Ctrl-c """
        try: 
            for epoch in range(num_epochs):
                for i, batch in enumerate(self.dataloader):
                    real_A = batch['A'].to(self.device)
                    real_B = batch['B'].to(self.device)
                    
                    losses = self.train_step(real_A, real_B)
                    
                    if i % 100 == 0:
                        print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(self.dataloader)}] "
                              f"[G loss: {losses['loss_G']:.4f}] [D loss: {losses['loss_D']:.4f}]")

                    if epoch % 5 == 0: 
                        torch.save({
                            'G_AB': self.G_AB.state_dict(),
                            'G_BA': self.G_BA.state_dict(),
                            'D_A': self.D_A.state_dict(),
                            'D_B': self.D_B.state_dict(),
                            'epoch': epoch
                        }, f'checkpoints/checkpoint_epoch_{epoch}.pth')
                        
        except KeyboardInterrupt:
            print("Training interrupted. Saving checkpoint...")
            torch.save({
                'G_AB': self.G_AB.state_dict(),
                'G_BA': self.G_BA.state_dict(),
                'D_A': self.D_A.state_dict(),
                'D_B': self.D_B.state_dict(),
                'epoch': epoch
            }, 'checkpoints/interrupted_checkpoint.pth') 
    
    def save_models(self, path):
        torch.save(self.G_AB.state_dict(), os.path.join(path, 'G_AB.pth'))
        torch.save(self.G_BA.state_dict(), os.path.join(path, 'G_BA.pth'))
        torch.save(self.D_A.state_dict(), os.path.join(path, 'D_A.pth'))
        torch.save(self.D_B.state_dict(), os.path.join(path, 'D_B.pth'))
    
    def load_models(self, path):
        self.G_AB.load_state_dict(torch.load(os.path.join(path, 'G_AB.pth')))
        self.G_BA.load_state_dict(torch.load(os.path.join(path, 'G_BA.pth')))
        self.D_A.load_state_dict(torch.load(os.path.join(path, 'D_A.pth')))
        self.D_B.load_state_dict(torch.load(os.path.join(path, 'D_B.pth')))


