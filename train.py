import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import json

def main():
    parser = argparse.ArgumentParser(description="Train a neural network for image classification")
    parser.add_argument('data_directory', metavar='DIR', help='path to the folder of flower images')
    parser.add_argument('--arch', default='vgg16', choices=['vgg16', 'densenet121'], help='pretrained model architecture: vgg16 or densenet121')
    parser.add_argument('--hidden_units', nargs='+', type=int, default=[512, 256, 128], help='list of sizes of hidden layers [512, 256, 128]')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate for training')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs for training')
    parser.add_argument('--gpu', action='store_true', help='use GPU for training since I have it available')

    args = parser.parse_args()

    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')

    print("Loading data...")
    data_loaders = load_data(args.data_directory)
    training_loader = data_loaders['training_loader']
    validation_loader = data_loaders['validation_loader']

    print("Creating model...")
    model = create_model(args.arch, args.hidden_units)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    print("Training model...")
    train_model(model, criterion, optimizer, training_loader, validation_loader, args.epochs, device)

    checkpoint_path = 'checkpoint.pth'
    print(f"Saving checkpoint to {checkpoint_path}...")
    save_checkpoint(model, optimizer, args.arch, args.hidden_units, args.epochs, checkpoint_path)

def load_data(data_dir):
    # Load the datasets with ImageFolder and define the transforms.
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define transforms for the training, validation, and testing sets
    training_transformations = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    validation_transformations = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    testing_transformations = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the datasets with ImageFolder
    training_dataset = datasets.ImageFolder(train_dir, transform=training_transformations)
    validation_dataset = datasets.ImageFolder(valid_dir, transform=validation_transformations)
    testing_dataset = datasets.ImageFolder(test_dir, transform=testing_transformations)

    # Using the image datasets and the trainforms, define the dataloaders
    training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=64, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=64)
    testing_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=64)

    return {
        'training_loader': training_loader,
        'validation_loader': validation_loader,
        'testing_loader': testing_loader
    }

def create_model(arch, hidden_units):
    # Load a pretrained network
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = 25088
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_size = 1024
    else:
        raise ValueError("Unsupported architecture. Only 'vgg16' or 'densenet121' are supported.")

    # Freeze model parameters to avoid backpropagation
    for param in model.parameters():
        param.requires_grad = False

    # Define a new classifier
    classifier = nn.Sequential(OrderedDict([
        ('dropout', nn.Dropout(0.2)),
        ('fc1', nn.Linear(input_size, hidden_units[0])),
        ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(hidden_units[0], hidden_units[1])),
        ('relu2', nn.ReLU()),
        ('fc3', nn.Linear(hidden_units[1], hidden_units[2])),
        ('relu3', nn.ReLU()),
        ('output', nn.Linear(hidden_units[2], 102)),
        ('log_softmax', nn.LogSoftmax(dim=1))
    ]))

    # Replace the classifier in the pretrained model
    if arch == 'vgg16':
        model.classifier = classifier
    elif arch == 'densenet121':
        model.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(1024, hidden_units[0])),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(hidden_units[0], hidden_units[1])),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(hidden_units[1], hidden_units[2])),
            ('relu3', nn.ReLU()),
            ('output', nn.Linear(hidden_units[2], 102)),
            ('log_softmax', nn.LogSoftmax(dim=1))
        ]))
    else:
        raise ValueError("Unsupported architecture. Only 'vgg16' or 'densenet121' are supported.")

    return model

def train_model(model, criterion, optimizer, training_loader, validation_loader, epochs, device):
    model.to(device)
    model.train()

    print('Training has started')

    print_every = 10
    steps = 0

    for epoch in range(epochs):
        running_loss = 0

        for ii, (training_images, training_labels) in enumerate(training_loader):
            steps += 1

            training_images, training_labels = training_images.to(device), training_labels.to(device)

            optimizer.zero_grad()

            outputs = model(training_images)
            loss = criterion(outputs, training_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                validation_loss = 0
                accuracy = 0

                with torch.no_grad():
                    for ii, (validation_images, validation_labels) in enumerate(validation_loader):
                        validation_images, validation_labels = validation_images.to(device), validation_labels.to(device)
                        log_ps = model(validation_images)
                        loss = criterion(log_ps, validation_labels)
                        validation_loss += loss.item()

                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equality = top_class == validation_labels.view(*top_class.shape)
                        accuracy += torch.mean(equality.type(torch.FloatTensor)).item()

                validation_loss /= len(validation_loader)
                accuracy /= len(validation_loader)

                print(f"Epoch: {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/len(training_loader):.3f}.. "
                      f"Test loss: {validation_loss:.3f}.. "
                      f"Test accuracy: {accuracy:.3f}")
                running_loss = 0
                model.train()
    print("\nTraining has ended")


def save_checkpoint(model, optimizer, arch, hidden_units, epochs, save_dir='.', filename='checkpoint.pth'):
    """Save the model checkpoint."""
    checkpoint = {
        'input_size': 25088 if arch == 'vgg16' else 1024,
        'output_size': 102,
        'hidden_units': hidden_units,
        'dropout': 0.2,
        'learning_rate': 0.001,
        'model_architecture': arch,
        'classifier': model.classifier,
        'model_state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'epochs': epochs,
        'optimizer_state_dict': optimizer.state_dict(),
    }

    checkpoint_path = os.path.join(save_dir, filename)
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

if __name__ == '__main__':
    main()