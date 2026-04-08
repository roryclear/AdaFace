from face_alignment import align
import numpy as np
import cv2
from PIL import Image
from datetime import datetime
from face_alignment import mtcnn
from blazeface import BlazeFace
import cv2
from tinygrad import Tensor


mtcnn_model = mtcnn.MTCNN(device='cpu', crop_size=(112, 112))


if __name__ == '__main__':
    blazeface = BlazeFace()

    img = cv2.imread("messi.jpg")
    img = Tensor(img)
    faces = blazeface(img)[0].numpy()
    print(faces)


    img = Image.open("messi.jpg").convert('RGB')
    _, faces = mtcnn_model.align_multi(img, limit=1)
    face = faces[0]
    aligned_rgb_img = np.array(face)

    cv2.imwrite("aligned_pic.jpg", cv2.cvtColor(aligned_rgb_img, cv2.COLOR_BGR2RGB))
    

