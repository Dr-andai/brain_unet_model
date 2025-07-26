import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torchvision.transforms as T
from unet import UNet

# Define model
model = UNet(in_channels=1, out_channels=3)
state_dict = torch.load("unet_epoch20.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

transform = T.Compose([
    T.Grayscale(),  # In case image is RGB
    T.Resize((256, 256)),
    T.ToTensor(),
])

# def predict(image: Image.Image):
#     img_tensor = transform(image).unsqueeze(0)
#     with torch.no_grad():
#         output = model(img_tensor)
#         pred = torch.argmax(F.softmax(output, dim=1), dim=1)
#     return pred.squeeze().numpy().tolist()  # list is JSON-serializable

def predict(image: Image.Image) -> Image.Image:
    input_tensor = transform(image).unsqueeze(0)  # (1, C, H, W)
    with torch.no_grad():
        output = model(input_tensor)[0]
    output = torch.argmax(output, dim=0).byte().cpu()
    output_pil = T.ToPILImage()(output)
    return output_pil