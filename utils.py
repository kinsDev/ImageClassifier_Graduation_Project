import torch
from torchvision import models
from PIL import Image
import numpy as np
import json

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model_architecture = checkpoint['model_architecture']
    
    if model_architecture == 'vgg16':
        model = models.vgg16(pretrained=False)
        input_size = 25088
    else:
        raise ValueError("Unsupported architecture. Only 'vgg16' is supported.")

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model

def process_image(image):
    # Define transformations
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = preprocess(image)
    return image

def load_class_names(category_names):
    with open(category_names, 'r') as f:
        class_to_name = json.load(f)
    return class_to_name