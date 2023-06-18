from copy import deepcopy

import torch
from PIL import Image
import pyscreenshot as ImageGrab
from torchvision import transforms

from network import LeNet5

# initializing model + transform + classes
model = LeNet5()
deepcopy(model.load_state_dict(torch.load("cnn.pt")))
model.eval()

trans = transforms.Compose([transforms.Resize((32, 32)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                 std=[0.5, 0.5, 0.5])
                            ])

classes = ["clear", "one", "two", "three", "four", "unopened"]

# screenshot of current frame
screenshot_region = (0, 344, 900, 1064)
img = ImageGrab.grab(bbox=screenshot_region)

# moving frame
board_map = []
y = 10
height = 82
for i in range(8):  # 8 is the num of rows
    x = 10
    width = 82
    row = []
    for j in range(10):  # 10 is the num of columns
        current_img = img.crop((x, y, width, height)).convert("RGB")
        with torch.no_grad():
            current_img = trans(current_img).to(torch.float32)
            current_img = current_img.unsqueeze(0)
            output = model(current_img)
            _, predicted = torch.max(output.data, 1)
            tile = classes[predicted.item()]
        if tile == "clear":
            row.append(0)
        elif tile == "one":
            row.append(1)
        elif tile == "two":
            row.append(2)
        elif tile == "three":
            row.append(3)
        elif tile == "four":
            row.append(4)
        elif tile == "unopened":
            row.append(5)

        x += (72 + 18)
        width += (72 + 18)
    board_map.append(row)
    y += (72 + 18)
    height += (72 + 18)

print()
for row in range(len(board_map)):
    for col in range(len(board_map[0])):
        print(board_map[row][col], end=" ")
    print()