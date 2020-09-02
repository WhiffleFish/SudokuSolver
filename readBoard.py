import cv2 as cv
import numpy as np
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks
import tensorflow as tf
from tensorflow import keras


def isolate_board(img):
    """
    INPUT:
        img -- Image containing a Sudoku board
    OUTPUT:
        img -- Image cropped to contain only the Sudoku board
    """
    edges = canny(img, sigma=1).astype(np.uint8)

    test_angles = np.linspace(-np.pi/2, np.pi/2, 360)  # Sweep -90° to +90°
    h, theta, d = hough_line(edges, theta=test_angles)

    top = img.shape[0] + 1
    bottom = -1
    left = img.shape[1] + 1
    right = -1

    # Horizontal Line: theta ~= pi/2
    # Vertical Line: theta ~= 0
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        # If line is Vertical
        if -45 < np.rad2deg(angle) < 45:
            # If line is furthest Left
            if dist < left: left = dist
            # If line is furthest Right
            if dist > right: right = dist
        else:  # Line is Horizontal
            # If line is furthest Down
            if dist > bottom: bottom = dist
            # If line is furthest Up
            if dist < top: top = dist

    return img[int(top):int(bottom), int(left):int(right)]


def read_board(img, model=None):
    """
    INPUTS:
        img -- Isolated image of sudoku board
        model -- Saved Keras CNN model to recognize digits
    OUTPUT:
        data -- numpy array of predicted Sudoku board
    """
    if not model:  # If model isn't specified beforehand
        model = keras.models.load_model(r"models\AUG_DigitRecognizerModel")

    def pred_num(img, thresh=0.5):
        '''
        Predict value of image of single number
        '''
        # resize image
        img_resized = cv.resize(img, (28, 28))

        # Change data shape and normalize
        img_formatted = img_resized.reshape(1, 28, 28, 1)/255

        # Make predictions
        pred_data = model.predict(img_formatted)[0]

        if pred_data.max() <= thresh:
            return 0
        else:
            return pred_data.argmax()

    dy, dx = np.array(img.shape)/9
    numbers = np.zeros((9, 9)).tolist()
    data = np.zeros((9, 9))

    for i in range(9):
        for j in range(9):
            number = img[int(dy*i):int(dy*(i+1)), int(dx*j):int(dx*(j+1))]
            numbers[i][j] = number
            data[i][j] = pred_num(number)

    return data


def isValid(puzzle, c):
    num = puzzle[c]

    puz = puzzle.copy()
    puz[c] = 0
    block = puz[(c[0]//3)*3 : ((c[0]+3)//3)*3, (c[1]//3)*3 : ((c[1]+3)//3)*3]

    # Check Row
    if num in puz[c[0]]:
        return False
    # Check Col
    elif num in puz[:,c[1]]:
        return False
    # Check Block
    elif num in block:
        return False
    else:
        return True


def solve_board(board):
    if 0 not in board:
        return board

    for i in range(9):
        for j in range(9):

            if board[i, j] == 0:
                for n in range(1, 10):
                    board[i, j] = n

                    if isValid(board, (i, j)):
                        solution = solve_board(board)
                        if solution is not None:
                            return solution

                # BACHTRACK
                board[i, j] = 0
                return None
