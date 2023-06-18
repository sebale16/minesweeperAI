import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader

from network import CustomDataset

all_transform = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                         std=[0.5, 0.5, 0.5])
                                    ])

test_dataset = CustomDataset("train_csv.csv", "train_pics/", transform=all_transform)
test_loader = DataLoader(test_dataset, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.load("cnn.pt")
# Function to test the model
running_accuracy = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        model.eval()
        inputs, outputs = data
        inputs, outputs = inputs.to(device), outputs.to(device)
        outputs = outputs.to(torch.float32)
        predicted_outputs = model(inputs)
        _, predicted = torch.max(predicted_outputs, 1)
        total += outputs.size(0)
        running_accuracy += (predicted == outputs).sum().item()

    print('Accuracy of the model based on the test set of', total,
          'inputs is: %d %%' % (100 * running_accuracy / total))
