from face_alignment import align
import numpy as np
import cv2
from PIL import Image
from datetime import datetime
from face_alignment import mtcnn
from blazeface import BlazeFace
import cv2
from tinygrad import Tensor
from mtcnn_pytorch.src.matlab_cp2tform import get_similarity_transform_for_cv2


mtcnn_model = mtcnn.MTCNN(device='cpu', crop_size=(112, 112))

REFERENCE_FACIAL_POINTS = [
    [38.29459953, 51.69630051],
    [73.53179932, 51.50139999],
    [56, 71.73660278],
    [56, 92.3655014 ]
]

if __name__ == '__main__':
    blazeface = BlazeFace()

    img = cv2.imread("img3.jpeg")
    img = Tensor(img)
    faces = blazeface(img)[0].numpy()

    # The ys are correct but not the Xs?
    facial5points = [
    [faces[4], faces[5]],   # left eye (kp1)
    [faces[6], faces[7]],   # right eye (kp0)
    [faces[8], faces[9]],   # nose (kp2)
    [faces[10], faces[11]], # mouth (kp3) → duplicate
    [faces[10], faces[11]]
    ]
    print(facial5points,"\n")


    img = Image.open("img3.jpeg").convert('RGB')
    boxes, faces = mtcnn_model.align_multi(img, limit=1)
    face = faces[0]
    aligned_rgb_img = np.array(face)

    cv2.imwrite("aligned_pic.jpg", cv2.cvtColor(aligned_rgb_img, cv2.COLOR_BGR2RGB))
    

