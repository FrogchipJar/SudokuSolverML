def is_valid(board, num, pos):
    """
    Checks if placing a number 'num' at a given position 'pos' on the board is valid
    according to Sudoku rules.

    Args:
        board (list of list of int): The 9x9 Sudoku board.
        num (int): The number to check.
        pos (tuple of int): The (row, col) position to check.

    Returns:
        bool: True if the move is valid, False otherwise.
    """
    row, col = pos

    # Rule 1: The number must not already exist in the current row.
    for i in range(len(board[0])):
        if board[row][i] == num and col != i:
            return False

    # Rule 2: The number must not already exist in the current column.
    for i in range(len(board)):
        if board[i][col] == num and row != i:
            return False

    # Rule 3: The number must not already exist in the current 3x3 sub-grid.
    box_x = col // 3
    box_y = row // 3

    for i in range(box_y * 3, box_y * 3 + 3):
        for j in range(box_x * 3, box_x * 3 + 3):
            if board[i][j] == num and (i, j) != pos:
                return False

    return True


def find_empty(board):
    """
    Finds the first empty cell (represented by a 0) in the Sudoku board.

    Args:
        board (list of list of int): The 9x9 Sudoku board.

    Returns:
        tuple of int: The (row, col) of the first empty cell, or None if the board is full.
    """
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == 0:
                return (i, j)  # (row, col)
    return None


def solve_sudoku(board):
    """
    Solves a Sudoku board in-place using the backtracking algorithm.

    This is a recursive function that works by trying to place numbers in empty
    cells one by one. If a number leads to a dead end, it backtracks and tries
    the next number.

    Args:
        board (list of list of int): The 9x9 Sudoku board to solve.

    Returns:
        bool: True if a solution was found, False otherwise.
    """
    # Find the next empty cell to work on.
    find = find_empty(board)
    if not find:
        return True  # Base case: If no empty cells are left, the puzzle is solved.
    else:
        row, col = find

    # Try placing numbers 1 through 9 in the empty cell.
    for num in range(1, 10):
        # Check if the number is valid in the current position.
        if is_valid(board, num, (row, col)):
            # If valid, place the number on the board.
            board[row][col] = num

            # Recursively call solve_sudoku to solve the rest of the puzzle.
            if solve_sudoku(board):
                return True

            # If the recursive call returns False, it means this path was a dead end.
            # Backtrack by resetting the cell to 0 and try the next number.
            board[row][col] = 0

    # If no number from 1-9 works in this cell, return False to trigger backtracking.
    return False