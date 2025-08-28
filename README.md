# SudokuSolverML

An AI agent that uses Python, a K-Nearest Neighbors (KNN) algorithm, and OpenCV to autonomously solve web-based Sudoku puzzles. It finds the grid on screen, reads the digits, solves the logic, and types the solution automatically.

## How It Works

The project is broken down into a few key steps:

1.  **Vision**: The script takes a screenshot of the screen and uses OpenCV to find the Sudoku grid. It then applies a perspective warp to get a flat, top-down view of the puzzle.
2.  **Digit Recognition**: The flattened grid is sliced into 81 individual cells. A K-Nearest Neighbors (KNN) model, trained on sample digit images, is used to identify the number in each cell.
3.  **Solving**: The recognized board is represented as a 2D array and solved using a standard backtracking algorithm.
4.  **Interaction**: The agent calculates the screen coordinates for each empty cell and uses `pyautogui` to click and type the solution numbers into the puzzle.

## Installation

To run this project, you'll need Python 3 and the dependencies listed in `requirements.txt`.

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd SudokuSolverML
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

## Usage

There are two main scripts you can run.

### Solving a Sudoku Puzzle

1.  Open a web-based Sudoku puzzle and make sure it is clearly visible on your screen.
2.  Run the main solver script:
    ```bash
    python main.py
    ```
3.  You will have 5 seconds to switch to the window containing the Sudoku puzzle. The agent will then take over.

### Collecting More Training Data

The accuracy of the digit recognition depends on the quality of the training data in the `train_data/` directory. If the model is failing to recognize certain digits, you can add more training samples using the `collect_data.py` script.

1.  Open a Sudoku puzzle.
2.  Run the data collection script:
    ```bash
    python collect_data.py
    ```
3.  The script will find the grid and show you each cell one by one.
    - Press the corresponding number key (`1`-`9`) to save the cell image as a training sample for that digit.
    - Press `s` to skip a cell.
    - Press `q` to quit.

## Configuration

### Mouse Click Accuracy (High-DPI Displays)

A critical setting for this project to work is the `SCALE_FACTOR` inside `interaction.py`.

-   **What it is:** This factor corrects for differences between the physical screen coordinates (used by OpenCV) and the logical screen coordinates (used by `pyautogui` for clicking).
-   **On macOS Retina displays:** This value is almost always `2`.
-   **On standard Windows/Linux displays:** This value is typically `1`.
-   **If clicks are off:** If the script is not clicking in the correct cells, you likely need to adjust this value. For example, if your display scaling in Windows is set to 150%, your `SCALE_FACTOR` would be `1.5`.
