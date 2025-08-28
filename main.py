import time
import sys

from vision import find_and_prepare_grid
from digit_recognizer import train_knn_model, build_sudoku_board
from solver import solve_sudoku
from interaction import enter_solution

def main():
    """
    Main function to run the Sudoku solver agent.

    This script orchestrates the entire process:
    1.  It gives the user time to switch to the Sudoku puzzle window.
    2.  It captures the screen and uses computer vision to find and isolate the Sudoku grid.
    3.  It trains a K-Nearest Neighbors (KNN) model on provided training data.
    4.  It uses the trained model to recognize the digits on the board.
    5.  It solves the puzzle using a backtracking algorithm.
    6.  Finally, it simulates keyboard input to enter the solution into the web page.
    """
    print("Starting Sudoku solver in 5 seconds...")
    print("Please switch to the window with the Sudoku puzzle.")
    time.sleep(5)

    # --- Part 1: Vision ---
    # This section handles finding the puzzle on the screen using our refactored helper.
    all_cells, grid_contour, _ = find_and_prepare_grid()
    if all_cells is None:
        # The error message is handled inside the helper function.
        sys.exit()

    # --- Part 2: Digit Recognition ---
    # This section identifies the numbers in the grid.
    print("\nStep 2: Training the digit recognition model...")
    try:
        knn_model = train_knn_model()
    except ValueError as e:
        print(f"Error during model training: {e}")
        sys.exit()

    print("Step 3: Building the digital representation of the board...")
    sudoku_board = build_sudoku_board(knn_model, all_cells)

    print("\nDetected Sudoku Board (0 represents an empty cell):")
    print(sudoku_board)
    print("-" * 25)

    # --- Part 3: Solving and Interaction ---
    # This section solves the puzzle and types the solution.
    print("Step 4: Solving the puzzle using a backtracking algorithm...")
    
    # Make a copy of the board to solve, preserving the original.
    solved_board = sudoku_board.copy()
    
    if solve_sudoku(solved_board):
        print("\nSolution Found:")
        print(solved_board)

        # The agent will now take control of the mouse and keyboard to enter the solution.
        enter_solution(sudoku_board, solved_board, grid_contour)
    else:
        print("\nCould not find a solution for the provided Sudoku puzzle.")



if __name__ == "__main__":
    main()