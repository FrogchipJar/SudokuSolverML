import pyautogui
import time
from vision import reorder_corners

def calculate_cell_center(grid_contour, row, col):
    """
    Calculates the screen coordinates of the center of a specific cell within the distorted grid.

    This function is crucial because the Sudoku grid on the screen might be tilted or viewed
    from an angle. A simple rectangular division would fail. Instead, this function uses the
    four detected corner points of the grid to map the (row, col) of a cell to its
    actual (x, y) pixel coordinates on the screen.

    Args:
        grid_contour: The 4 corner points of the Sudoku grid found by OpenCV.
        row: The cell's row index (0-8).
        col: The cell's column index (0-8).

    Returns:
        A tuple (x, y) of the screen coordinates for the cell's center.
    """
    # Reorder the corners into a consistent order: top-left, top-right, bottom-right, bottom-left.
    corners = reorder_corners(grid_contour)
    (tl, tr, br, bl) = corners

    # To find the center of a cell, we interpolate its position based on the grid's corners.
    # This accounts for any perspective distortion.

    # Calculate the position of the cell's top edge by interpolating along the top edge of the grid.
    top_x = tl[0] + (col / 9.0) * (tr[0] - tl[0])
    top_y = tl[1] + (col / 9.0) * (tr[1] - tl[1])
    
    # Calculate the position of the cell's bottom edge by interpolating along the bottom edge of the grid.
    bottom_x = bl[0] + (col / 9.0) * (br[0] - bl[0])
    bottom_y = bl[1] + (col / 9.0) * (br[1] - bl[1])
    
    # The X-coordinate of the cell's center is the average of its top and bottom edge positions.
    center_x = (top_x + bottom_x) / 2.0
    
    # Similarly, find the Y-coordinate by interpolating along the left and right edges.
    left_y = tl[1] + (row / 9.0) * (bl[1] - tl[1])
    right_y = tr[1] + (row / 9.0) * (br[1] - tr[1])
    center_y = (left_y + right_y) / 2.0

    # The above gives the top-left corner of the cell. To find the center, we add
    # half the cell's width and height.
    cell_width = ((tr[0] - tl[0]) + (br[0] - bl[0])) / 18.0  # Avg width of grid / 9 cells
    cell_height = ((bl[1] - tl[1]) + (br[1] - tr[1])) / 18.0 # Avg height of grid / 9 cells

    return int(center_x + cell_width / 2.0), int(center_y + cell_height / 2.0)


def enter_solution(original_board, solved_board, grid_contour):
    """
    Takes control of the mouse and keyboard to type the solution into the Sudoku puzzle.

    It iterates through the board and, for each cell that was originally empty, it calculates
    the cell's center on the screen, clicks it, and types the correct number.
    """
    print("\nStep 5: Entering the solution...")
    print("IMPORTANT: The script will now control your mouse and keyboard. Do not interfere.")
    
    # =======================================================================================
    # !!! CRITICAL: HIGH-DPI (RETINA) DISPLAY CONFIGURATION !!!
    # =======================================================================================
    # PyAutoGUI uses logical screen coordinates, while OpenCV uses physical pixel coordinates.
    # On high-DPI displays (like Apple's Retina screens), these are not the same.
    # The SCALE_FACTOR bridges this gap.
    #
    # - For macOS Retina Displays: This value is almost always 2.
    # - For standard displays (most Windows/Linux): This should be 1.
    #
    # If clicks are inaccurate, you may need to find your OS's display scale factor.
    # For example, in Windows display settings, if you have "Scale and layout" set to 150%,
    # your scale factor would be 1.5.
    SCALE_FACTOR = 2
    # =======================================================================================

    time.sleep(2)

    for row in range(9):
        for col in range(9):
            # Only type in cells that were originally empty.
            if original_board[row][col] == 0:
                # 1. Calculate the PHYSICAL pixel coordinates where the cell is on the screen.
                physical_x, physical_y = calculate_cell_center(grid_contour, row, col)
                
                # 2. Convert to LOGICAL screen coordinates that pyautogui can use.
                logical_x = physical_x / SCALE_FACTOR
                logical_y = physical_y / SCALE_FACTOR
                
                # 3. Click the cell and type the number.
                pyautogui.click(logical_x, logical_y)
                pyautogui.typewrite(str(solved_board[row][col]), interval=0.01)

    print("\n--- Solution Entered! Project Complete! ---")