from transformers import AutoModel, AutoConfig
from PIL import Image
import torch
from torchvision import transforms
from unet import UNet

# Define model
# model = UNet(in_channels=1, out_channels=3)
# state_dict = torch.load("unet_epoch20.pth", map_location="cpu")
# model.load_state_dict(state_dict)
# model.eval()

config = AutoConfig.from_pretrained(".")
model = AutoModel.from_pretrained(".", config=config)
model.eval()

# Define preprocessing
preprocess = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((config.image_size, config.image_size)),
    transforms.ToTensor(),
])

# Define inference function
def predict(image: Image.Image) -> Image.Image:
    input_tensor = preprocess(image).unsqueeze(0)  # Shape: (1, C, H, W)
    with torch.no_grad():
        output = model(input_tensor).squeeze(0)     # Shape: (3, H, W)
        prediction = torch.argmax(output, dim=0).byte()  # Shape: (H, W)
    return transforms.ToPILImage()(prediction)