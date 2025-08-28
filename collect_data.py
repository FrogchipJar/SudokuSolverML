import cv2
import os
import sys
import time

from vision import find_and_prepare_grid, strip_borders

def main():
    """
    Main function to run the data collection utility.

    This script allows a user to easily create training data for the KNN model.
    It performs the same vision pipeline as the main solver to find the Sudoku
    grid and isolate the 81 cells. It then displays each cell one by one and
    prompts the user to press the corresponding number key (1-9) to save it.
    """
    print("Starting data collector in 5 seconds...")
    print("Please switch to the window with the Sudoku puzzle.")
    time.sleep(5)

    # --- Part 1: Vision Pipeline ---
    # Use the refactored helper function to find and process the grid.
    all_cells, _, _ = find_and_prepare_grid()
    if all_cells is None:
        sys.exit()

    # --- Part 2: Data Collection Loop ---
    print("\n--- Starting Data Collector ---")
    print("For each cell, press the number key (1-9) to save it as training data.")
    print("Press 's' to skip the current cell.")
    print("Press 'q' to quit the data collector.")

    # Create the base training data directory if it doesn't exist.
    if not os.path.exists('train_data'):
        os.makedirs('train_data')

    for i, cell in enumerate(all_cells):
        # Strip the border to get a clean image of the digit.
        cell_for_display = strip_borders(cell, border_size=2)

        # Show the current cell to the user.
        cv2.imshow("Collect Training Data - Press key (1-9), 's' to skip, 'q' to quit", cv2.resize(cell_for_display, (200, 200)))

        key = cv2.waitKey(0)

        if key == ord('q'):
            print("Quitting data collection.")
            break
        elif key == ord('s'):
            print(f"Skipping cell {i}.")
            continue
        # Check if the key pressed is a digit from '1' to '9'.
        elif ord('1') <= key <= ord('9'):
            digit = chr(key)

            # Define the path to save the image.
            save_path = os.path.join('train_data', digit)

            # Create the specific digit's directory if it doesn't exist.
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # Save the image with a unique name to avoid overwriting.
            count = len(os.listdir(save_path)) + 1
            filename = f"cell_{i}_{count}.png"
            # Use the original, un-stripped cell for saving to be consistent with preprocessing
            cell_to_save = strip_borders(cell)
            cv2.imwrite(os.path.join(save_path, filename), cell_to_save)
            print(f"Saved cell {i} as a '{digit}' in '{save_path}'")

    cv2.destroyAllWindows()
    print("\n--- Data Collection Complete ---")

if __name__ == "__main__":
    main()
