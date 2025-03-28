import cv2
import numpy as np

def create_checkerboard(squares_x=9, squares_y=6, square_size=100):
    """Create a checkerboard pattern image and save it for printing"""
    # Create a blank image with white background
    board_width = squares_x * square_size
    board_height = squares_y * square_size
    checkerboard = np.ones((board_height, board_width), dtype=np.uint8) * 255
    
    # Fill with black squares in a checkerboard pattern
    for y in range(squares_y):
        for x in range(squares_x):
            if (x + y) % 2 == 0:
                y1 = y * square_size
                y2 = (y + 1) * square_size
                x1 = x * square_size
                x2 = (x + 1) * square_size
                checkerboard[y1:y2, x1:x2] = 0
    
    # Save the checkerboard image
    cv2.imwrite("checkerboard_to_print.png", checkerboard)
    print(f"Checkerboard created and saved as 'checkerboard_to_print.png'")
    
    # Display the checkerboard
    cv2.imshow("Checkerboard Pattern", checkerboard)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Create standard 9x6 checkerboard
create_checkerboard(9, 6, 100)