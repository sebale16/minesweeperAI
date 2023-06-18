from copy import deepcopy

import torch.nn as nn

# Load in relevant libraries, and alias where appropriate
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_df = pd.DataFrame(columns=["img_path", "label"])
train_df["img_path"] = os.listdir("train_pics/")
for idx, i in enumerate(os.listdir("train_pics/")):
    if "clear" in i:
        train_df["label"][idx] = 0
    if "one" in i:
        train_df["label"][idx] = 1
    if "two" in i:
        train_df["label"][idx] = 2
    if "three" in i:
        train_df["label"][idx] = 3
    if "four" in i:
        train_df["label"][idx] = 4
    if "unopened" in i:
        train_df["label"][idx] = 5

train_df.to_csv(r'train_csv.csv', index=False, header=True)

class CustomDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idex):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idex, 0])
        image = Image.open(img_path).convert("RGB")
        labl = self.img_labels.iloc[idex, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            labl = self.target_transform(labl)
        return image, labl


# Define relevant variables for the ML task
batch_size = 8
learning_rate = 0.001
num_epochs = 20

all_transform = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                         std=[0.5, 0.5, 0.5])
                                    ])

train_dataset = CustomDataset("train_csv.csv", "train_pics/", transform=all_transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)


# Defining the convolutional neural network
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, 6)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out


model = LeNet5().to(device)

# Setting the loss function
cost = nn.CrossEntropyLoss()

# Setting the optimizer with the model parameters and learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# this is defined to print how many steps are remaining when training
total_step = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = cost(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.7f}'
              .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

print('Finished Training')
torch.save(deepcopy(model.state_dict()), "cnn.pt")