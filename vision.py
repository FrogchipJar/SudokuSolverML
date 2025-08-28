import cv2
import numpy as np
import pyautogui

def capture_and_preprocess():
    """
    Takes a screenshot, converts it to grayscale, blurs, and applies adaptive threshold.
    """
    image = pyautogui.screenshot()
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return thresh, image

def find_grid_contour(processed_image):
    """
    Finds and returns the largest contour with 4 corners, likely the Sudoku grid.
    """
    contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for c in sorted_contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx
    return None

def reorder_corners(points):
    """
    Reorders the four corner points into a consistent order: 
    top-left, top-right, bottom-right, bottom-left.
    """
    points = points.reshape((4, 2))
    rect = np.zeros((4, 2), dtype="float32")
    
    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    
    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    
    return rect

def warp_perspective(image, corners):
    """
    Takes the original image and corner points, and returns a flat, top-down view.
    """
    ordered_corners = reorder_corners(corners)
    (tl, tr, br, bl) = ordered_corners

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    destination_points = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    matrix = cv2.getPerspectiveTransform(ordered_corners, destination_points)
    warped_image = cv2.warpPerspective(image, matrix, (maxWidth, maxHeight))
    return warped_image

def extract_cells(flattened_grid):
    """
    Slices the flattened grid image into an array of 81 cell images.
    """
    cells = []
    cell_size = 50 # Assuming a 450x450 grid
    for i in range(9):
        for j in range(9):
            y_start, y_end = i * cell_size, (i + 1) * cell_size
            x_start, x_end = j * cell_size, (j + 1) * cell_size
            cells.append(flattened_grid[y_start:y_end, x_start:x_end])
    return np.array(cells)

def strip_borders(cell_image, border_size=4):
    """
    Strips the outer border from a cell image by cropping it.
    This is more robust than masking.
    
    Args:
        cell_image: The image of a single Sudoku cell.
        border_size: How many pixels to shave off from each side.
        
    Returns:
        A smaller image containing only the center of the cell.
    """
    # Get the dimensions of the image
    h, w = cell_image.shape
    
    # Crop the image by slicing it.
    # [startY:endY, startX:endX]
    cropped_cell = cell_image[border_size : h - border_size, border_size : w - border_size]
    
    return cropped_cell