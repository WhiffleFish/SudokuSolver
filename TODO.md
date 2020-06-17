# TODO
- Operate in venv
- Isolate Board for oblique view angle
- Account for page curvature 
    - straight cell boundaries become curved
    - Map upper cell boundary to straight line

# IDEAS
- https://aishack.in/tutorials/sudoku-grabber-opencv-detection/
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