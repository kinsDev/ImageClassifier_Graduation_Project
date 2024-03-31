import argparse
import torch
import numpy as np 
from torch import nn, optim
from torchvision import models
from PIL import Image
import json

def main():
    parser = argparse.ArgumentParser(description="Predict flower0 name from an image")
    parser.add_argument('image_path', metavar='PATH', help='path to the image file')
    parser.add_argument('checkpoint', metavar='CHECKPOINT', help='path to the model checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='mapping of categories to real names')
    parser.add_argument('--gpu', action='store_true', help='use GPU for inference since I have it available')

    args = parser.parse_args()

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    print("Loading checkpoint...")
    model = load_checkpoint(args.checkpoint)
    model.to(device)
    model.eval()

    print("Predicting...")
    probabilities, classes = predict(args.image_path, model, device, args.top_k)

    print("Mapping classes to names...")
    class_to_name = load_class_names(args.category_names)
    class_names = [class_to_name[class_] for class_ in classes]

    print("Results:")
    for i in range(len(probabilities)):
        print(f"{class_names[i]}: {probabilities[i]:.3f}")

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg16(pretrained=False) 
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

from PIL import Image
import numpy as np
import torch
from torchvision import transforms

def process_image(image):
    # Define the transformations
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = preprocess(image)
    np_image = img.numpy()
    return np_image

def predict(image_path, model, device, topk=5):
    # Open the image using PIL
    with Image.open(image_path) as image:
        # Preprocess the image
        image = process_image(image)
        image = torch.from_numpy(image).float().to(device)
        image = image.unsqueeze(0)

    # Set the model to evaluation mode
    model.eval()
    
    # Move the model to the same device as the input tensor
    model = model.to(device)

    # Perform inference
    with torch.no_grad():
        output = model(image)

    probabilities, indices = torch.topk(torch.exp(output), topk)
    probabilities = probabilities.squeeze().tolist()
    indices = indices.squeeze().tolist()
    
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    classes = [idx_to_class[idx] for idx in indices]

    return probabilities, classes


def load_class_names(category_names):
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f, strict = False) # To avoid an error at some workspaces and library versions
    return cat_to_name


if __name__ == '__main__':
    main()