import pyautogui
import torch
import pyscreeze as ImageGrab
from torchvision import transforms
from time import sleep

from network import LeNet5

def click(_m, _n, _i, _j, bomb_map, board_map, mouse_map):
    if _m != 0 and _n != 0:
        if bomb_map[_i + _m][_j + _n] == 0 and board_map[_i + _m][_j + _n] == 5:
            pyautogui.click(mouse_map[_i + _m][_j + _n][0], mouse_map[_i + _m][_j + _n][1])

# initializing model + transform + classes
def main():
    model = LeNet5()
    model.load_state_dict(torch.load("cnn.pth", map_location=torch.device('cpu')))
    model.eval()

    trans = transforms.Compose([transforms.Resize((32, 32)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                     std=[0.5, 0.5, 0.5])
                                ])

    # initialize bomb map + bomb count
    y = 773+45
    bomb_map = []
    mouse_map = []
    for i in range(8):
        x = 1150+45
        mouse_row = []
        bomb_row = []
        for j in range(10):
            mouse_row.append([x, y])
            bomb_row.append(0)
            x += 90
        mouse_map.append(mouse_row)
        bomb_map.append(bomb_row)
        y += 90

    bombs_found = 0
    sleep(2)
    pyautogui.click(1550, 1175)

    # while not all bombs have been found
    while bombs_found < 10:

        # screenshot of current frame
        screenshot_region = (1150, 773, 900, 719)
        img = ImageGrab.screenshot(region=screenshot_region)

        # moving frame + reading map
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
                    row.append(predicted.item())

                x += (72 + 18)
                width += (72 + 18)
            board_map.append(row)
            y += (72 + 18)
            height += (72 + 18)

        # looping through inner grid
        for i in range(8):
            for j in range(10):
                tile = board_map[i][j]
                unopened_tile_count = 0
                bombs_around_count = 0
                unopened_tiles_but_bomb_x = None
                unopened_tiles_but_bomb_y = None
                unopened_tiles = []
                unopened_tiles_but_no_bomb_xs = []
                unopened_tiles_but_no_bomb_ys = []
                if i == 0:
                    if j == 0:
                        for k in (0, 1):
                            for l in (0, 1):
                                if k == 0 and l == 0:
                                    continue
                                else:
                                    if bomb_map[i + k][j + l] == 1:
                                        bombs_around_count += 1
                                    if board_map[i + k][j + l] == 5:
                                        unopened_tile_count += 1
                                        if bomb_map[i + k][j + l] == 0:
                                            unopened_tiles.append([mouse_map[i + k][j + l][0], mouse_map[i + k][j + l][1]])
                                            unopened_tiles_but_no_bomb_xs.append(k)
                                            unopened_tiles_but_no_bomb_ys.append(l)
                                        unopened_tiles_but_bomb_x = k
                                        unopened_tiles_but_bomb_y = l
                    elif j == 9:
                        for k in (0, 1):
                            for l in (-1, 0):
                                if k == 0 and l == 0:
                                    continue
                                else:
                                    if bomb_map[i + k][j + l] == 1:
                                        bombs_around_count += 1
                                    if board_map[i + k][j + l] == 5:
                                        unopened_tile_count += 1
                                        if bomb_map[i + k][j + l] == 0:
                                            unopened_tiles.append(
                                                [mouse_map[i + k][j + l][0], mouse_map[i + k][j + l][1]])
                                            unopened_tiles_but_no_bomb_xs.append(k)
                                            unopened_tiles_but_no_bomb_ys.append(l)
                                        unopened_tiles_but_bomb_x = k
                                        unopened_tiles_but_bomb_y = l
                    else:
                        for k in (0, 1):
                            for l in (-1, 0, 1):
                                if k == 0 and l == 0:
                                    continue
                                else:
                                    if bomb_map[i + k][j + l] == 1:
                                        bombs_around_count += 1
                                    if board_map[i + k][j + l] == 5:
                                        unopened_tile_count += 1
                                        if bomb_map[i + k][j + l] == 0:
                                            unopened_tiles.append(
                                                [mouse_map[i + k][j + l][0], mouse_map[i + k][j + l][1]])
                                            unopened_tiles_but_no_bomb_xs.append(k)
                                            unopened_tiles_but_no_bomb_ys.append(l)
                                        unopened_tiles_but_bomb_x = k
                                        unopened_tiles_but_bomb_y = l
                elif i == 7:
                    if j == 0:
                        for k in (-1, 0):
                            for l in (0, 1):
                                if k == 0 and l == 0:
                                    continue
                                else:
                                    if bomb_map[i + k][j + l] == 1:
                                        bombs_around_count += 1
                                    if board_map[i + k][j + l] == 5:
                                        unopened_tile_count += 1
                                        if bomb_map[i + k][j + l] == 0:
                                            unopened_tiles.append(
                                                [mouse_map[i + k][j + l][0], mouse_map[i + k][j + l][1]])
                                            unopened_tiles_but_no_bomb_xs.append(k)
                                            unopened_tiles_but_no_bomb_ys.append(l)
                                        unopened_tiles_but_bomb_x = k
                                        unopened_tiles_but_bomb_y = l
                    elif j == 9:
                        for k in (-1, 0):
                            for l in (-1, 0):
                                if k == 0 and l == 0:
                                    continue
                                else:
                                    if bomb_map[i + k][j + l] == 1:
                                        bombs_around_count += 1
                                    if board_map[i + k][j + l] == 5:
                                        unopened_tile_count += 1
                                        if bomb_map[i + k][j + l] == 0:
                                            unopened_tiles.append(
                                                [mouse_map[i + k][j + l][0], mouse_map[i + k][j + l][1]])
                                            unopened_tiles_but_no_bomb_xs.append(k)
                                            unopened_tiles_but_no_bomb_ys.append(l)
                                        unopened_tiles_but_bomb_x = k
                                        unopened_tiles_but_bomb_y = l
                    else:
                        for k in (-1, 0):
                            for l in (-1, 0, 1):
                                if k == 0 and l == 0:
                                    continue
                                else:
                                    if bomb_map[i + k][j + l] == 1:
                                        bombs_around_count += 1
                                    if board_map[i + k][j + l] == 5:
                                        unopened_tile_count += 1
                                        if bomb_map[i + k][j + l] == 0:
                                            unopened_tiles.append(
                                                [mouse_map[i + k][j + l][0], mouse_map[i + k][j + l][1]])
                                            unopened_tiles_but_no_bomb_xs.append(k)
                                            unopened_tiles_but_no_bomb_ys.append(l)
                                        unopened_tiles_but_bomb_x = k
                                        unopened_tiles_but_bomb_y = l
                else:
                    if j == 0:
                        for k in (-1, 0, 1):
                            for l in (0, 1):
                                if k == 0 and l == 0:
                                    continue
                                else:
                                    if bomb_map[i + k][j + l] == 1:
                                        bombs_around_count += 1
                                    if board_map[i + k][j + l] == 5:
                                        unopened_tile_count += 1
                                        if bomb_map[i + k][j + l] == 0:
                                            unopened_tiles.append(
                                                [mouse_map[i + k][j + l][0], mouse_map[i + k][j + l][1]])
                                            unopened_tiles_but_no_bomb_xs.append(k)
                                            unopened_tiles_but_no_bomb_ys.append(l)
                                        unopened_tiles_but_bomb_x = k
                                        unopened_tiles_but_bomb_y = l
                    elif j == 9:
                        for k in (-1, 0, 1):
                            for l in (-1, 0):
                                if k == 0 and l == 0:
                                    continue
                                else:
                                    if bomb_map[i + k][j + l] == 1:
                                        bombs_around_count += 1
                                    if board_map[i + k][j + l] == 5:
                                        unopened_tile_count += 1
                                        if bomb_map[i + k][j + l] == 0:
                                            unopened_tiles.append(
                                                [mouse_map[i + k][j + l][0], mouse_map[i + k][j + l][1]])
                                            unopened_tiles_but_no_bomb_xs.append(k)
                                            unopened_tiles_but_no_bomb_ys.append(l)
                                        unopened_tiles_but_bomb_x = k
                                        unopened_tiles_but_bomb_y = l
                    else:
                        for k in range(-1, 2):
                            for l in range(-1, 2):
                                if k == 0 and l == 0:
                                    continue
                                else:
                                    if bomb_map[i + k][j + l] == 1:
                                        bombs_around_count += 1
                                    if board_map[i + k][j + l] == 5:
                                        unopened_tile_count += 1
                                        if bomb_map[i + k][j + l] == 0:
                                            unopened_tiles.append(
                                                [mouse_map[i + k][j + l][0], mouse_map[i + k][j + l][1]])
                                            unopened_tiles_but_no_bomb_xs.append(k)
                                            unopened_tiles_but_no_bomb_ys.append(l)
                                        unopened_tiles_but_bomb_x = k
                                        unopened_tiles_but_bomb_y = l

                if tile == 1:
                    if bombs_around_count == 0:
                        if unopened_tile_count == 1:
                            if bomb_map[i + unopened_tiles_but_bomb_x][j + unopened_tiles_but_bomb_y] != 1:
                                bomb_map[i + unopened_tiles_but_bomb_x][j + unopened_tiles_but_bomb_y] = 1
                                bombs_found += 1
                    else:
                        if unopened_tile_count > 1:
                            for t in range(len(unopened_tiles)):
                                pyautogui.click(unopened_tiles[t][0], unopened_tiles[t][1])

                elif 1 < tile < 5:
                    if bombs_around_count < tile:
                        if unopened_tile_count == tile:
                            for t in range(len(unopened_tiles_but_no_bomb_xs)):
                                if bomb_map[i + unopened_tiles_but_no_bomb_xs[t]][j + unopened_tiles_but_no_bomb_ys[t]] != 1:
                                    bomb_map[i + unopened_tiles_but_no_bomb_xs[t]][j + unopened_tiles_but_no_bomb_ys[t]] = 1
                                    bombs_around_count += 1
                                    bombs_found += 1
                    if tile == bombs_around_count:
                        if i == 0:
                            if j == 0:
                                for m in (0, 1):
                                    for n in (0, 1):
                                        click(m, n, i, j, bomb_map, board_map, mouse_map)
                            elif j == 9:
                                for m in (0, 1):
                                    for n in (0, -1):
                                        click(m, n, i, j, bomb_map, board_map, mouse_map)

                            else:
                                for m in (0, 1):
                                    for n in (-1, 0, 1):
                                        click(m, n, i, j, bomb_map, board_map, mouse_map)

                        if i == 7:
                            if j == 0:
                                for m in (0, -1):
                                    for n in (0, 1):
                                        click(m, n, i, j, bomb_map, board_map, mouse_map)

                            elif j == 9:
                                for m in (0, -1):
                                    for n in (0, -1):
                                        click(m, n, i, j, bomb_map, board_map, mouse_map)

                            else:
                                for m in (0, -1):
                                    for n in (-1, 0, 1):
                                        click(m, n, i, j, bomb_map, board_map, mouse_map)

                        else:
                            if j == 0:
                                for m in (-1, 0, 1):
                                    for n in (0, 1):
                                        click(m, n, i, j, bomb_map, board_map, mouse_map)

                            elif j == 9:
                                for m in (-1, 0, 1):
                                    for n in (0, -1):
                                        click(m, n, i, j, bomb_map, board_map, mouse_map)

                            else:
                                for m in range(-1, 2):
                                    for n in range(-1, 2):
                                        click(m, n, i, j, bomb_map, board_map, mouse_map)

    for row in range(len(board_map)):
        for col in range(len(board_map[0])):
            if board_map[row][col] == 5 and bomb_map[row][col] == 0:
                pyautogui.click(mouse_map[row][col][0], mouse_map[row][col][1])

if __name__ == '__main__':
    main()
