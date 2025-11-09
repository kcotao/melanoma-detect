import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 3
CLASS_NAMES = ["melanoma", "nervus", "seborrheic_keratosis"]  # orden de tu dataset

def build_resnet18(num_classes: int = NUM_CLASSES) -> nn.Module:
    # sin pesos ImageNet (para evitar conflictos de shape al cargar state_dict)
    model = models.resnet18(weights=None)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    return model

def load_model_from_state(path: str, device: str = "cpu") -> nn.Module:
    model = build_resnet18(NUM_CLASSES).to(device)
    sd = torch.load(path, map_location=device)
    # Permite cargar si el state_dict está dentro de un dict
    state_dict = sd["model_state"] if isinstance(sd, dict) and "model_state" in sd else sd
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model