import cv2
import numpy as np

def get_region(image, mode = "peghole"):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        green_box_contour = max(contours, key=cv2.contourArea)
        box_mask = np.zeros_like(green_mask)
        cv2.drawContours(box_mask, [green_box_contour], -1, 255, thickness=cv2.FILLED)
    else: # in some peg cases
        print("No green region found. Using an empty box mask.")
        box_mask = np.zeros_like(green_mask)

    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 70])
    dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)
    black_arrow_mask = cv2.bitwise_and(dark_mask, box_mask)

    if mode == "peg":
        black_arrow_mask = cv2.bitwise_and(dark_mask, cv2.bitwise_not(box_mask))

    output = np.zeros_like(image)
    output[black_arrow_mask == 255] = [255, 255, 255]

    return output
