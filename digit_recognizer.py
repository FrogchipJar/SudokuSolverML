import cv2
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from vision import strip_borders

def preprocess_digit(cell_image):
    """
    Processes a single cell image to prepare it for the KNN model.

    This involves:
    1.  Stripping the border to remove grid lines.
    2.  Using Otsu's thresholding to create a clean binary image of the digit.
        Otsu's method is effective because it automatically determines the best
        threshold value, making it robust to variations in lighting.
    3.  Finding the bounding box of the digit to crop out empty space.
    4.  Resizing the cropped digit to a standard 20x20 pixel image.
    5.  Flattening the 2D image into a 1D array of 400 pixels for the KNN model.
    """
    # Strip the outer border to remove any grid lines that may interfere.
    # A slightly larger border size is used for safety.
    stripped = strip_borders(cell_image, border_size=6)
    
    # Use Otsu's thresholding. It automatically finds the optimal threshold
    # to separate the digit from the background. This is better than a fixed
    # threshold (e.g., 128) because it adapts to lighting changes.
    _, thresh = cv2.threshold(stripped, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find the contour of the digit to get its bounding box.
    coords = cv2.findNonZero(thresh)
    if coords is None:
        # If there are no white pixels, it's an empty cell.
        return np.zeros((20*20), dtype=np.uint8)
    
    x, y, w, h = cv2.boundingRect(coords)
    digit = thresh[y:y+h, x:x+w]
    
    # Resize to the standard 20x20 size used for training.
    resized = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_AREA)
    
    # Flatten the 20x20 image into a 400-element vector for the model.
    return resized.flatten()

def train_knn_model(data_path='train_data/'):
    """
    Loads image data from the 'train_data' directory and trains a K-Nearest Neighbors (KNN) model.

    The 'train_data' directory should contain subdirectories named '1', '2', ..., '9',
    each filled with example images of that digit.
    """
    samples = []
    labels = []

    # Loop through each digit's training data directory.
    for digit in range(1, 10):
        digit_path = os.path.join(data_path, str(digit))
        if not os.path.isdir(digit_path):
            print(f"Warning: Training data directory not found for digit '{digit}' at {digit_path}")
            continue
        
        print(f"Loading samples for digit: {digit}")
        for filename in os.listdir(digit_path):
            # Load the image in grayscale and process it just like the real input.
            img = cv2.imread(os.path.join(digit_path, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                samples.append(preprocess_digit(img))
                labels.append(digit)

    if not labels:
        raise ValueError("No training data was found. Ensure the 'train_data' directory"
                         " is populated with digit images in subfolders '1' through '9'.")

    # Create and train the KNN model. n_neighbors=3 is a common starting point.
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(np.array(samples, dtype=np.float32), labels)
    print("KNN Model trained successfully!")
    return model


def is_cell_empty(cell_image, contour_threshold=15):
    """
    Determines if a Sudoku cell is empty or contains a digit.

    This works by thresholding the image and checking if any significant contours
    (i.e., shapes) are present. Small noise is ignored.
    
    Args:
        cell_image: A grayscale image of a single Sudoku cell.
        contour_threshold: The minimum pixel area required to be considered a digit.
                           Anything smaller is treated as noise.
        
    Returns:
        True if the cell is likely empty, False otherwise.
    """
    # First, strip the outer borders to remove grid lines.
    stripped = strip_borders(cell_image)
    
    # **Consistency Improvement**: Use Otsu's thresholding here as well.
    # This makes the empty-cell check robust to the same lighting variations
    # that the digit pre-processing is.
    _, thresh = cv2.threshold(stripped, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours in the thresholded image.
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If no contours are found at all, the cell is definitely empty.
    if not contours:
        return True
        
    # If contours are found, check if the largest one is substantial enough to be a digit.
    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) > contour_threshold:
        return False  # A significant contour was found, so it's a digit.
        
    # If all contours are smaller than the threshold, it's likely just noise.
    return True


def build_sudoku_board(model, all_cells):
    """
    Constructs the 9x9 Sudoku board by classifying the digit in each cell image.

    It iterates through the 81 cell images, determines if each is empty or contains
    a digit, and builds a NumPy array representing the board.
    """
    board = np.zeros((9, 9), dtype=int)
    
    for i, cell in enumerate(all_cells):
        # First, check if the cell is empty to avoid unnecessary predictions.
        if is_cell_empty(cell):
            digit = 0
        else:
            # If not empty, process the cell and predict the digit using the trained model.
            processed_cell = preprocess_digit(cell)
            digit = model.predict([processed_cell])[0]

        # Place the recognized digit onto the board at the correct position.
        row, col = i // 9, i % 9
        board[row, col] = digit
        
    return board
