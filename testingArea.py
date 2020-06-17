import cv2 as cv
import matplotlib.pyplot as plt

from readBoard import isolate_board

img = cv.imread(r'images\sudoku2.png', cv.IMREAD_GRAYSCALE)

board = isolate_board(img)
fig, ax = plt.subplots(1,2)
ax[0].imshow(img, cmap='gist_gray')
ax[1].imshow(board, cmap='gist_gray')
plt.show()
