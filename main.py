import cv2
import time
import sys, os

from vision import capture_and_preprocess, find_grid_contour, warp_perspective, extract_cells, strip_borders
from digit_recognizer import train_knn_model, build_sudoku_board, preprocess_digit, is_cell_empty
from solver import solve_sudoku
from interaction import enter_solution

def main():
    """
    Main function to run the Sudoku solver agent.
    """
    print("Starting Sudoku solver in 5 seconds...")
    time.sleep(5)

    # --- 1. VISION PIPELINE ---
    print("Step 1: Capturing and processing the screen...")
    processed_image, original_screenshot = capture_and_preprocess()

    print("Step 2: Finding the Sudoku grid...")
    grid_contour = find_grid_contour(processed_image)

    if grid_contour is None:
        print("Error: Sudoku grid not found on the screen. Exiting.")
        sys.exit()

    flattened_grid = warp_perspective(original_screenshot, grid_contour)
    flattened_grid = cv2.resize(flattened_grid, (450, 450))
    
    gray_flattened_grid = cv2.cvtColor(flattened_grid, cv2.COLOR_BGR2GRAY)
    all_cells = extract_cells(gray_flattened_grid)
    print("Grid found and sliced into 81 cells.")


    # --- 2. DIGIT RECOGNITION PIPELINE ---
    print("Step 3: Training the digit recognition model...")
    try:
        knn_model = train_knn_model()
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit()


    # # =================================================================
    # # DATA COLLECTOR LOOP
    # # =================================================================
    # print("\n--- Starting Data Collector ---")
    # print("For each cell, press the number key (1-9) to save it as training data.")
    # print("Press 's' to skip a cell. Press 'q' to quit.")

    # for i, cell in enumerate(all_cells):
    #     cell = strip_borders(cell)
    #     # Show the current cell
    #     cv2.imshow("Collect Training Data", cv2.resize(cell, (200, 200)))
        
    #     key = cv2.waitKey(0)
        
    #     if key == ord('q'):
    #         break
    #     elif key == ord('s'):
    #         continue
    #     # Check if the key pressed is a digit from '1' to '9'
    #     elif ord('1') <= key <= ord('9'):
    #         digit = chr(key)
    #         # Create the directory if it doesn't exist
    #         save_path = os.path.join('train_data', digit)
    #         if not os.path.exists(save_path):
    #             os.makedirs(save_path)
            
    #         # Save the image with a unique name
    #         count = len(os.listdir(save_path)) + 1
    #         filename = f"cell_{i}_{count}.png"
    #         cv2.imwrite(os.path.join(save_path, filename), cell)
    #         print(f"Saved cell {i} as a '{digit}'")

    # cv2.destroyAllWindows()
    # # =================================================================

    print("Step 4: Building the digital board...")
    sudoku_board = build_sudoku_board(knn_model, all_cells)

    print("\nDetected Sudoku Board:")
    print(sudoku_board)
    print("-" * 25)

    # --- 3. SOLVER & INTERACTION ---
    print("Step 5: Solving the puzzle...")
    
    # Make a copy so the original board is preserved
    solved_board = sudoku_board.copy()
    
    if solve_sudoku(solved_board):
        print("\nSolved Board:")
        print(solved_board)
        # debug_coordinates(original_screenshot, grid_contour)
        enter_solution(sudoku_board, solved_board, grid_contour)
    else:
        print("\nCould not find a solution for the puzzle.")



    # # =================================================================
    # # FINAL CLASSIFICATION DEBUGGER
    # # =================================================================
    # print("\n--- Starting Classification Debugger ---")
    # print("For each NON-EMPTY cell, you will see the original vs. the processed version.")
    # print("Press any key to advance. Press 'q' to quit.")

    # # We need the trained model for this
    # knn_model = train_knn_model() 

    # for i, cell in enumerate(all_cells):
    #     # Only run the debugger on cells that your function thinks are NOT empty
    #     if not is_cell_empty(cell):
    #         print("-" * 30)
    #         print(f"Analyzing Cell #{i} (which the code thinks is a digit)...")

    #         # --- Show the Original ---
    #         cv2.imshow("1. Original Cell", cv2.resize(cell, (200, 200)))
            
    #         # --- Process and Predict ---
    #         processed_data = preprocess_digit(cell)
    #         prediction = knn_model.predict([processed_data])[0]
            
    #         # --- Show the Processed Version ---
    #         # Reshape the 400-pixel array back into a 20x20 image to display it
    #         processed_img = processed_data.reshape((20, 20))
    #         cv2.imshow("2. Processed Digit (What the Model Sees)", 
    #                 cv2.resize(processed_img, (200, 200), interpolation=cv2.INTER_NEAREST))
            
    #         print(f"MODEL PREDICTION: {prediction}")
            
    #         key = cv2.waitKey(0)
    #         if key == ord('q'):
    #             break
                
    # cv2.destroyAllWindows()
    # # =================================================================



if __name__ == "__main__":
    main()