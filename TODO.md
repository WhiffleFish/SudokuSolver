# TODO
- Operate in venv
- Capture 70+ images of Sudoku boards on paper in different environments
- Label captured images for semantic segmentation
- Isolate Board for oblique view angle
- Account for page curvature 
    - straight cell boundaries become curved
    - Map upper cell boundary to straight line

# IDEAS
- Board grab through floodfill
    - https://aishack.in/tutorials/sudoku-grabber-opencv-detection/
- Augmented Reality on Sudoku Puzzle usingComputer Vision and Deep Learning
    - https://www.ijitee.org/wp-content/uploads/papers/v8i11S2/K102209811S219.pdf
- MATLAB Semantic Segmentation 
    - https://blogs.mathworks.com/deep-learning/2018/11/15/sudoku-solver-image-processing-and-deep-learning/
- OpenCV Perspective Transformation
    - https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html 

## For actual pictures of Sudoku puzzle
- HAAR Cascade?
- Draw bounding lines around all cells
    - Linear Transform on image so that bouding lines form square
    - https://www.geeksforgeeks.org/line-detection-python-opencv-houghline-method/

# DONE
- Isolate Board for direct view angle (DONE)
- Augment mnist picture data to account for bars from cell frame (DONE)
- Fill out requirements (DONE)
- Canny ED Image before Hough Transform (DONE)
    - Cell edges found regardless of whether they contrast by being relatively light or dark