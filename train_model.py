import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from data_loader import *
from model import *
from Solver import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from util.data_def import *


def main():
    
    # Generate a folder that saves model
    if not os.path.exists(save_model):
        os.mkdir(save_model)
        
        
    epochs = 10
    learning_rate = 1e-3
    batch_size = 32
    
    transform = transforms.Compose([transforms.ToTensor(),
                    transforms.Resize((224, 224)),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])])

    image_set = SegmentationData(txt_file, img_path, transform)

    # Split the dataset into training set and testing set
    data_num = len(image_set)
    training_num = int(0.8 * data_num)
    testing_num = data_num - training_num
    training_set, testing_set = random_split(image_set, [training_num, testing_num])

    # data loader
    train_dataloader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(testing_set, batch_size=10, shuffle=True)

    # model
    device = ("cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu")
  
    print(f"Using {device} device")

    model = SegmentationNN().to(device)
    solver = Solver(model, epochs, train_dataloader, device, learning_rate)

    # train the model
    solver.train()
    
    # model_path = current_directory + '/save_model/seg_network_9.pt'
    # model.load_state_dict(torch.load(model_path))
    # model.eval()

    # Testing accuracy

    total_samples = 0
    total_correct = 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    with torch.no_grad():
        for i, (image, label) in enumerate(tqdm(test_dataloader)):

            image = image.to(device)
            label = label.to(device)

            outputs = model(image.float())
            _, pred = torch.max(outputs, 1)

            total_correct += (pred == label).sum().item()
            total_samples += label.size(0)

    print("\nTesting Accuracy {}%".format(int(100 * total_correct / total_samples)))
    
  
if __name__ == "__main__":
    main()