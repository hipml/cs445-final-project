import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import LBFGS 

import torchvision.models as models
from torchvision.models import VGG19_Weights
import torchvision.transforms as transforms
from torchvision.utils import save_image

from tqdm import tqdm
from PIL import Image

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()

        # let's get the convolutional layers after MaxPool layers
        self.select_features = ['0', '5', '10', '19', '28']
        self.vgg = models.vgg19(weights=VGG19_Weights.DEFAULT).features


    def forward(self, output):
        features = []
        for name, layer in self.vgg._modules.items():
            output = layer(output)
            if name in self.select_features:
                features.append(output)
        return features

# content loss function
def get_content_loss(target, content):
    """
    Calculates the loss between content image and target image
    
    L_content(p, x, l) = 1/2 * (sum (F_ij - P_ij, ij)^2)
    """
    return (0.5 * torch.mean((target - content)**2))

# style loss function
# equivalent to computing the maximum mean discrepancy between two images
def get_style_loss(target, style):
    _, c, h, w = target.size()

    # calculate Gram matrix for target image
    G = gram_matrix(target)

    # calculate Gram matrix for style image
    S = gram_matrix(style)

    return torch.mean((G - S)**2)



# gram matrix is calculated for every layer
def gram_matrix(input):
    """
    Compute the Gram matrix for each layer

    G_ij = sum(F_ik * F_jk, k)
    """

    batch_size, channels, height, width = input.size()
    features = input.view(batch_size * channels, height * width)
    gram = torch.mm(features, features.t())
    return gram.div(channels * height * width)

def load_img(path, loader):
    img = Image.open(path)
    img = loader(img).unsqueeze(0)
    return img

def save(target, i):
    denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
    img = target.clone().squeeze()
    img = denorm(img).clamp(0, 1)
    save_image(img, f'result_{i}.png')

def main():
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        vgg = VGG().to(device).eval()

        img_size = 512 if torch.cuda.is_available() else 128

        # normalizing using imagenet mean and stddev values for RGB
        loader = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        steps = 2000
        alpha = 1 # content weight
        beta = 1e7 #style weight

        #content_img = load_img('images/input/chimi.jpg', loader).to(device)
        content_img = load_img('images/input/input.jpg', loader).to(device)
        # style_img = load_img('images/art/vangogh/self-portrait_1998.74.5.jpg', loader).to(device)
        style_img = load_img('images/art/vangogh/farmhouse_in_provence_1970.17.34.jpg', loader).to(device)
        # style_img = load_img('images/art/monet/banks_of_the_seine,_vetheuil_1963.10.177.jpg', loader).to(device)

        # initialize output to be a random noise image
        # target_img = torch.randn_like(content_img).to(device).requires_grad_(True)
        # optimizer = optim.Adam([target_img], lr=1e-3)

        # LBFGS is used in the original paper
        target_img = content_img.clone().requires_grad_(True)
        optimizer = LBFGS([target_img], max_iter=1, line_search_fn='strong_wolfe')
        iteration = [0]
        def closure():
            optimizer.zero_grad()

            # get features
            target_features = vgg(target_img)
            content_features = vgg(content_img)
            style_features = vgg(style_img)

            # calculate losses
            style_loss = 0
            content_loss = 0

            for target, content, style in zip(target_features, content_features, style_features):
                content_loss += get_content_loss(target, content)
                style_loss += get_style_loss(target, style)

            # calculate total loss according to paper
            total_loss = alpha * content_loss + beta * style_loss

            # set parameters to zero, compute gradient, update parameters
            total_loss.backward()

            if iteration[0] % 50 == 0:
                print(f'step: {iteration[0]}, content loss: {content_loss.item()}, style loss: {style_loss.item()}')
            if iteration[0] %100 == 0:
                save(target_img, iteration[0])

            iteration[0] += 1

            return total_loss

        for _ in range(1000):
            optimizer.step(closure)
        
        save(target_img, "final")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        torch.cuda.empty_cache()
    
if __name__ == "__main__":
    main()
