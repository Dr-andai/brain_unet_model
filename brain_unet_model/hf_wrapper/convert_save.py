import torch
from model import UNetTransformerModel, UNetConfig

# Load config and model
config = UNetConfig(in_channels=1, out_channels=3, image_size=256)
model = UNetTransformerModel(config)

# Load your existing model weights
model.model.load_state_dict(torch.load("unet_epoch20.pth", map_location="cpu"))

# Save the model and config in HF-compatible format
model.save_pretrained("brain_unet_hf")
config.save_pretrained("brain_unet_hf")

print("✅ Saved to brain_unet_hf/")
