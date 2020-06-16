import cv2 as cv
import logging
import numpy as np
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks
import tensorflow as tf
from tensorflow import keras


def isolate_board(img):
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


def check_row(board, pos):
    num = board[pos[0]][pos[1]]
    row = board[pos[0]].copy()
    row.pop(pos[1])
    if num in row:
        logging.debug('ROW FAIL')
    return num not in row


def check_col(board, pos, debug=False):
    num = board[pos[0]][pos[1]]
    col = [row[pos[1]] for row in board.copy()]
    col.pop(pos[0])
    if num in col:
        logging.debug('COL FAIL')
    return num not in col


def check_block(board, pos, debug=False):
    # Get current Number
    num = board[pos[0]][pos[1]]
    # Get upper left block corner
    block_start = ((int(pos[0]/3)*3), (int(pos[1]/3))*3)
    # Get lower right block corner
    block_end = (block_start[0]+3, block_start[1]+3)

    block = [row[block_start[1]:block_end[1]] for row in board[block_start[0]:block_end[0]].copy()]
    flat_block = [num for row in block for num in row]
    rel_coord = (pos[0]-block_start[0], pos[1]-block_start[1])
    rel_index = rel_coord[0]*3 + rel_coord[1]
    assert flat_block[rel_index] == num
    flat_block.pop(rel_index)
    if num in flat_block:
        logging.debug('BLOCK FAIL')
    return num not in flat_block


def check_num(board, pos):
    return check_row(board, pos) and check_col(board, pos) and check_block(board, pos)


def solve_board(board, c=(0, 0)):
    # Find next zero
    current_pos = c[0]*9 + c[1]
    flattened = [num for row in board for num in row]

    while True:
        if flattened[current_pos] == 0:
            break
        current_pos += 1
        if current_pos == 9*9:  # No More Zeros
            return board

    current_coord = (int(current_pos/9), current_pos % 9)

    while True:
        board[current_coord[0]][current_coord[1]] += 1

        logging.debug(np.array(board))
        if check_num(board, current_coord):
            resulting_board = solve_board(board, current_coord)
            flattened_result_board = [num for row in resulting_board for num in row]
            if not (0 in flattened_result_board):
                return resulting_board
        if board[current_coord[0]][current_coord[1]] >= 9:
            board[current_coord[0]][current_coord[1]] = 0
            logging.debug("RESET")
            return board
