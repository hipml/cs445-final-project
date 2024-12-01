import torch
from torchvision import transforms
from PIL import Image
from cyclegan import Generator, CycleGAN  

def style_transfer(image_path, model_path, output_path):
    # load model
    generator = Generator().to('cuda')
    generator.load_state_dict(torch.load(f'{model_path}/G_AB.pth', weights_only=True))
    generator.eval()

    # load and preprocess image
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to('cuda')
    
    with torch.no_grad():
        styled_image = generator(image)
    
    # convert to PIL image and save
    output = transforms.ToPILImage()(styled_image[0] * 0.5 + 0.5)
    output.save(output_path)

cyclegan = CycleGAN(
    root_A='images/input',
    root_B='images/art/monet',
    device='cuda'
)

#cyclegan.train(num_epochs=40)
#cyclegan.save_models('checkpoints/monet')

style_transfer('images/input/chimi-look.jpg', 'checkpoints/monet', 'output_monet.jpg')
