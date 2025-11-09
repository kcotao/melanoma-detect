from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import pillow_heif
pillow_heif.register_heif_opener()

# Misma transformación que se usó para entrenar el modelo
TF = transforms.Compose([
    transforms.Resize(340, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def preprocess_pil(img: Image.Image) -> torch.Tensor:
    if img.mode != "RGB":
        img = img.convert("RGB")
    x = TF(img).unsqueeze(0)  # (1,3,300,300)
    return x
