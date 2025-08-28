import cv2
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from vision import strip_borders

def preprocess_digit(cell_image):
    """
    Processes a single cell image for the ML model using Otsu's thresholding
    to handle anti-aliased digits correctly.
    """
    # First, strip the outer border to remove grid lines.
    # A slightly larger border size is safer.
    stripped = strip_borders(cell_image, border_size=6)
    
    # --- THE KEY CHANGE IS HERE ---
    # Instead of a hardcoded 128, we use cv2.THRESH_OTSU.
    # The '0' is a placeholder; Otsu's method calculates the real threshold.
    # We combine it with THRESH_BINARY_INV.
    _, thresh = cv2.threshold(stripped, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # -----------------------------
    
    # Find the bounding box that contains all white pixels (the digit)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return np.zeros((20*20), dtype=np.uint8)
    
    x, y, w, h = cv2.boundingRect(coords)
    digit = thresh[y:y+h, x:x+w]
    
    # Resize to our standard 20x20 size
    resized = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_AREA)
    
    return resized.flatten()

def train_knn_model(data_path='train_data/'):
    """
    Loads training data and trains a KNN model.
    """
    samples = []
    labels = []

    for digit in range(1, 10):
        digit_path = os.path.join(data_path, str(digit))
        if not os.path.isdir(digit_path):
            print(f"Warning: Directory not found for digit {digit} at {digit_path}")
            continue
        
        print(f"Loading samples for digit: {digit}")
        for filename in os.listdir(digit_path):
            img = cv2.imread(os.path.join(digit_path, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                samples.append(preprocess_digit(img))
                labels.append(digit)

    if not labels:
        raise ValueError("No training data found. Have you created the 'train_data' directory and populated it with digit images?")

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(np.array(samples, dtype=np.float32), labels)
    print("KNN Model trained successfully!")
    return model


def is_cell_empty(cell_image, contour_threshold=15):
    """
    Checks if a cell is empty by looking for contours of a minimum size.
    
    Args:
        cell_image: A grayscale image of a single Sudoku cell.
        contour_threshold: The minimum pixel area to be considered a digit.
        
    Returns:
        True if the cell is likely empty, False otherwise.
    """
    # First, strip the outer borders to remove grid lines
    stripped = strip_borders(cell_image)
    
    # Threshold the image to get a binary representation
    _, thresh = cv2.threshold(stripped, 128, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If no contours are found, the cell is definitely empty
    if not contours:
        return True
        
    # If contours are found, check if the largest one is big enough to be a digit
    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) > contour_threshold:
        return False # A significant contour was found, so it's not empty
        
    # If all contours are smaller than the threshold, it's likely just noise
    return True


def build_sudoku_board(model, all_cells):
    """
    Identifies the digit in each cell and reconstructs the 9x9 board.
    """
    board = np.zeros((9, 9), dtype=int)
    
    for i, cell in enumerate(all_cells):
        if is_cell_empty(cell):
            digit = 0
        else:
            processed_cell = preprocess_digit(cell)
            digit = model.predict([processed_cell])[0]

        # Place the digit on the board
        row, col = i // 9, i % 9
        board[row, col] = digit
        
    return board
