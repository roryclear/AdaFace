from face_alignment import align
import numpy as np
import cv2


if __name__ == '__main__':
    aligned_rgb_img = align.get_aligned_face("messi.jpg")
    cv2.imwrite("aligned_pic.jpg", cv2.cvtColor(aligned_rgb_img, cv2.COLOR_BGR2RGB))
    

