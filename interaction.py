import pyautogui
import cv2
import time
from vision import reorder_corners

def calculate_cell_center(grid_contour, row, col):
    """
    Calculates the screen coordinates of the center of a specific cell.
    
    Args:
        grid_contour: The 4 corner points of the Sudoku grid found by OpenCV.
        row: The cell's row index (0-8).
        col: The cell's column index (0-8).
        
    Returns:
        A tuple (x, y) of the screen coordinates.
    """
    # Reorder the corners into a consistent order: tl, tr, br, bl
    corners = reorder_corners(grid_contour)
    (tl, tr, br, bl) = corners

    # Calculate the position of the cell by interpolating between the corners.
    # This is more robust than just dividing the width.
    
    # Position along the top edge
    top_x = tl[0] + (col / 9.0) * (tr[0] - tl[0])
    top_y = tl[1] + (col / 9.0) * (tr[1] - tl[1])
    
    # Position along the bottom edge
    bottom_x = bl[0] + (col / 9.0) * (br[0] - bl[0])
    bottom_y = bl[1] + (col / 9.0) * (br[1] - bl[1])
    
    # Position along the left edge
    left_x = tl[0] + (row / 9.0) * (bl[0] - tl[0])
    left_y = tl[1] + (row / 9.0) * (bl[1] - tl[1])
    
    # Position along the right edge
    right_x = tr[0] + (row / 9.0) * (br[0] - tr[0])
    right_y = tr[1] + (row / 9.0) * (br[1] - tr[1])
    
    # The cell's x is the average of its position on the top and bottom edges
    # The cell's y is the average of its position on the left and right edges
    cell_x = (top_x + bottom_x) / 2
    cell_y = (left_y + right_y) / 2
    
    # We want the center, so we add half a cell's width/height
    cell_width = ((tr[0] - tl[0]) + (br[0] - bl[0])) / 18.0 # Average width / 9
    cell_height = ((bl[1] - tl[1]) + (br[1] - tr[1])) / 18.0 # Average height / 9
    
    center_x = cell_x + (cell_width / 2)
    center_y = cell_y + (cell_height / 2)

    return int(center_x), int(center_y)


# In interaction.py

def enter_solution(original_board, solved_board, grid_contour):
    """
    Types the solution into the Sudoku puzzle on the screen.
    """
    print("\nStep 6: Entering the solution...")
    print("IMPORTANT: Do not move the mouse or use the keyboard.")
    
    # =========================================================================
    # !!! IMPORTANT: SCALE FACTOR FOR HIGH-DPI/RETINA DISPLAYS !!!
    # On macOS Retina displays, this value is almost always 2.
    # On standard displays (most Windows/Linux), it should be 1.
    # If clicks are still off, you may need to find your OS's specific scale factor.
    SCALE_FACTOR = 2 
    # =========================================================================

    time.sleep(2)

    for row in range(9):
        for col in range(9):
            if original_board[row][col] == 0:
                # 1. Calculate the PHYSICAL pixel coordinates from OpenCV
                physical_x, physical_y = calculate_cell_center(grid_contour, row, col)
                
                # 2. Convert to LOGICAL coordinates for pyautogui
                logical_x = physical_x / SCALE_FACTOR
                logical_y = physical_y / SCALE_FACTOR
                
                # 3. Perform the actions using the corrected logical coordinates
                pyautogui.click(logical_x, logical_y)
                pyautogui.typewrite(str(solved_board[row][col]), interval=0.01)

    print("\n--- Solution Entered! Project Complete! ---")


# def debug_coordinates(original_screenshot, grid_contour):
#     """
#     A visual tool to verify the grid contour and calculated cell centers
#     before any clicking occurs.
#     """
#     if grid_contour is None:
#         print("DEBUG ERROR: The grid_contour passed to the function is None!")
#         return

#     print("\n--- Starting Coordinate Debugger ---")
    
#     # Create a color copy of the screenshot to draw on
#     debug_image = original_screenshot.copy()

#     # --- 1. Visualize the Grid Contour ---
#     print(f"DEBUG: Contour points received: \n{grid_contour}")
#     cv2.drawContours(debug_image, [grid_contour], -1, (0, 255, 0), 3) # Draw a thick green box

#     # --- 2. Visualize the First and Last Cell Centers ---
#     # This checks if the coordinate calculation is reasonable.
#     first_cell_x, first_cell_y = calculate_cell_center(grid_contour, 0, 0) # Cell at (0,0)
#     last_cell_x, last_cell_y = calculate_cell_center(grid_contour, 8, 8)  # Cell at (8,8)
    
#     print(f"DEBUG: Calculated center for cell (0,0) is ({first_cell_x}, {first_cell_y})")
#     print(f"DEBUG: Calculated center for cell (8,8) is ({last_cell_x}, {last_cell_y})")
    
#     # Draw circles on the calculated centers
#     cv2.circle(debug_image, (first_cell_x, first_cell_y), 10, (255, 0, 0), -1) # Blue circle for start
#     cv2.circle(debug_image, (last_cell_x, last_cell_y), 10, (0, 0, 255), -1)   # Red circle for end

#     # --- 3. Display the Result ---
#     print("Look at the 'Coordinate Debugger' window.")
#     print(" - Is the green box tightly around the Sudoku grid?")
#     print(" - Are the blue and red circles inside the grid?")
#     print("Press 'q' to continue.")
    
#     cv2.imshow("Coordinate Debugger", debug_image)
#     key = cv2.waitKey(0)
#     cv2.destroyAllWindows()