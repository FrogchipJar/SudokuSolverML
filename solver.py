# solver.py

def is_valid(board, num, pos):
    """
    Checks if placing a number 'num' at a given position 'pos' is valid.
    pos is a tuple (row, col).
    """
    row, col = pos

    # Check row
    for i in range(len(board[0])):
        if board[row][i] == num and col != i:
            return False

    # Check column
    for i in range(len(board)):
        if board[i][col] == num and row != i:
            return False

    # Check 3x3 box
    box_x = col // 3
    box_y = row // 3

    for i in range(box_y * 3, box_y * 3 + 3):
        for j in range(box_x * 3, box_x * 3 + 3):
            if board[i][j] == num and (i, j) != pos:
                return False

    return True


def find_empty(board):
    """
    Finds the first empty cell (represented by 0) in the board.
    """
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == 0:
                return (i, j)  # row, col
    return None


def solve_sudoku(board):
    """
    Solves a Sudoku board using the backtracking algorithm.
    The function modifies the board in-place.
    Returns True if a solution is found, False otherwise.
    """
    find = find_empty(board)
    if not find:
        return True  # Puzzle is already solved
    else:
        row, col = find

    for num in range(1, 10):
        if is_valid(board, num, (row, col)):
            board[row][col] = num

            # Recursive call
            if solve_sudoku(board):
                return True

            # Backtrack: if the recursive call failed, reset the cell
            board[row][col] = 0

    return False