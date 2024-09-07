# Minesweeper AI
## Demo
![](https://github.com/sebale16/minesweeperAI/blob/main/minesweeper_bot_demo1.gif)
## How it works
The bot takes a screenshot of every tile on the board, runs it through the trained CNN, and outputs to a 2d array representing the rows and columns of the board.
Then, the algorithm computes which mines are bombs by reading the marker present on an open tile and counting the number of unopened tiles around it. 
If they are equal, then all those unopened tiles are bombs, and they are mapped to a separate 2d array.
The algorithm then tells the mouse to click on the tiles that are known to not have bombs, i.e. bombs around a marker are already marked and there are more unopened tiles than the marker count.
Sometimes, though, this algorithm fails and the bot gets stuck, so user input is required.
