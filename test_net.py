import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader

from network import CustomDataset, LeNet5

all_transform = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                         std=[0.5, 0.5, 0.5])
                                    ])

test_dataset = CustomDataset("train_csv.csv", "train_pics/", transform=all_transform)
test_loader = DataLoader(test_dataset, shuffle=False)

model = LeNet5()
model.load_state_dict(torch.load("cnn.pth"))
model.eval()

# test the model
running_accuracy = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        model.eval()
        inputs, outputs = data
        outputs = outputs.to(torch.float32)
        predicted_outputs = model(inputs)
        _, predicted = torch.max(predicted_outputs, 1)
        total += outputs.size(0)
        running_accuracy += (predicted == outputs).sum().item()

    print('Accuracy of the model based on the test set of', total,
          'inputs is: %d %%' % (100 * running_accuracy / total))
