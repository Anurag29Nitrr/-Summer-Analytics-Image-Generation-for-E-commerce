# product_image_augmentation.py

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
from models import Generator  # Assuming a predefined Generator class for GAN

# Load the pretrained GAN model (replace with the actual path to the model)
model_path = 'path/to/pretrained_gan.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = Generator().to(device)
generator.load_state_dict(torch.load(model_path, map_location=device))
generator.eval()

# Transformations for the input image
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Function to generate augmented images
def generate_augmented_images(image_path, num_images=5):
    original_image = Image.open(image_path).convert('RGB')
    input_image = transform(original_image).unsqueeze(0).to(device)
    
    augmented_images = []
    for _ in range(num_images):
        noise = torch.randn(1, 100, 1, 1, device=device)
        generated_image = generator(noise)
        augmented_images.append(generated_image)
    
    # Save augmented images
    for idx, img in enumerate(augmented_images):
        save_image(img, f'augmented_image_{idx + 1}.png', normalize=True)

if __name__ == "__main__":
    # Example image path
    image_path = 'path/to/product_image.jpg'
    
    # Generate and save augmented images
    generate_augmented_images(image_path)
