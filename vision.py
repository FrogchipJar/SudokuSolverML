import cv2
import numpy as np
import pyautogui

# --- Constants ---
# Define constants for grid dimensions to avoid "magic numbers".
GRID_SIZE = 450
CELL_SIZE = GRID_SIZE // 9

def capture_and_preprocess():
    """
    Captures the screen and applies a series of image processing steps to
    prepare it for grid detection.

    Returns:
        A tuple containing:
        - The final processed (thresholded) image, ready for contour detection.
        - The original, unprocessed screenshot (in BGR color format).
    """
    # Take a screenshot and convert it from an array to the BGR format OpenCV uses.
    image = pyautogui.screenshot()
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Convert to grayscale, as color is not needed for grid detection.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply a Gaussian blur to reduce noise and improve thresholding.
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Use adaptive thresholding to create a binary image. This is better than a simple
    # global threshold because it can handle variations in lighting across the image.
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    return thresh, image

def find_grid_contour(processed_image):
    """
    Finds the Sudoku grid in a processed image by identifying the largest
    quadrilateral contour.

    Args:
        processed_image: A binary image that has been preprocessed.

    Returns:
        The contour (a set of points) corresponding to the Sudoku grid,
        or None if no suitable contour is found.
    """
    # Find all external contours in the image.
    contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area in descending order to prioritize larger shapes.
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Iterate through contours to find one that is a quadrilateral.
    for c in sorted_contours:
        peri = cv2.arcLength(c, True)
        # Approximate the contour shape to a simpler polygon.
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # A Sudoku grid is a quadrilateral, so it should have 4 corner points.
        if len(approx) == 4:
            return approx

    return None

def reorder_corners(points):
    """
    Reorders the four corner points of a quadrilateral into a consistent sequence:
    top-left, top-right, bottom-right, and bottom-left.

    This is necessary for applying a perspective transform correctly. The method works
    by observing the sum and difference of the corner coordinates.

    Args:
        points: A NumPy array of four corner points.

    Returns:
        A NumPy array of the same points in a standardized order.
    """
    points = points.reshape((4, 2))
    rect = np.zeros((4, 2), dtype="float32")
    
    # The top-left point will have the smallest sum (x+y).
    # The bottom-right point will have the largest sum (x+y).
    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    
    # The top-right point will have the smallest difference (y-x).
    # The bottom-left point will have the largest difference (y-x).
    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    
    return rect

def warp_perspective(image, corners):
    """
    Applies a perspective warp to the image to get a flat, top-down view of the Sudoku grid.

    Args:
        image: The original, un-processed image.
        corners: The four corner points of the Sudoku grid.

    Returns:
        A new image showing the rectified, top-down view of the grid.
    """
    ordered_corners = reorder_corners(corners)
    (tl, tr, br, bl) = ordered_corners

    # Calculate the width of the new image, which will be the
    # maximum distance between bottom-right/bottom-left and top-right/top-left x-coordinates.
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Calculate the height of the new image, which will be the
    # maximum distance between top-right/bottom-right and top-left/bottom-left y-coordinates.
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Define the destination points for the top-down view.
    destination_points = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Compute the perspective transform matrix and apply it.
    matrix = cv2.getPerspectiveTransform(ordered_corners, destination_points)
    warped_image = cv2.warpPerspective(image, matrix, (maxWidth, maxHeight))
    return warped_image

def extract_cells(flattened_grid):
    """
    Slices the flattened Sudoku grid image into an array of 81 individual cell images.
    Uses the GRID_SIZE and CELL_SIZE constants for calculations.
    """
    cells = []
    for i in range(9):
        for j in range(9):
            y_start, y_end = i * CELL_SIZE, (i + 1) * CELL_SIZE
            x_start, x_end = j * CELL_SIZE, (j + 1) * CELL_SIZE
            cells.append(flattened_grid[y_start:y_end, x_start:x_end])
    return np.array(cells)

def find_and_prepare_grid():
    """
    A helper function that encapsulates the entire vision pipeline.
    It captures, finds, processes, and extracts all cells from a Sudoku grid on screen.

    Returns:
        A tuple containing (all_cells, grid_contour, original_screenshot) if successful,
        or (None, None, None) if the grid cannot be found.
    """
    print("Capturing the screen and finding the Sudoku grid...")
    processed_image, original_screenshot = capture_and_preprocess()
    grid_contour = find_grid_contour(processed_image)

    if grid_contour is None:
        print("Error: Sudoku grid not found on the screen. Make sure it is visible.")
        return None, None, None

    # Once the grid is found, warp it to a flat, top-down view.
    flattened_grid = warp_perspective(original_screenshot, grid_contour)
    flattened_grid = cv2.resize(flattened_grid, (GRID_SIZE, GRID_SIZE))

    # Convert to grayscale and slice the grid into 81 individual cell images.
    gray_flattened_grid = cv2.cvtColor(flattened_grid, cv2.COLOR_BGR2GRAY)
    all_cells = extract_cells(gray_flattened_grid)
    print("Grid found and sliced into 81 cells.")

    return all_cells, grid_contour, original_screenshot

def strip_borders(cell_image, border_size=4):
    """
    Strips a border of a specified size from a cell image by cropping it.

    This is used to remove the grid lines from each cell before digit recognition,
    which helps prevent the lines from being mistaken for part of a digit.
    
    Args:
        cell_image: The image of a single Sudoku cell.
        border_size: The number of pixels to shave off from each of the four sides.
        
    Returns:
        A smaller, cropped image containing only the center of the cell.
    """
    # Get the height and width of the image.
    h, w = cell_image.shape
    
    # Crop the image by slicing it. The format is [startY:endY, startX:endX].
    cropped_cell = cell_image[border_size : h - border_size, border_size : w - border_size]
    
    return cropped_cell