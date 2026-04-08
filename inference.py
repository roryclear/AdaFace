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

def img_to_face(orig):
    h, w = orig.shape[:2]
    scale = 640 / max(h, w)
    resized = cv2.resize(orig, (int(w*scale), int(h*scale)))
    delta_w, delta_h = 640 - resized.shape[1], 640 - resized.shape[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    orig = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
    detections = blazeface(Tensor(orig)).numpy()
    detections = detections[detections[:, 4] != 0]
    if detections.shape[0] > 0:
        x1, y1, x2, y2 = detections[0][:4]
        detections[0][4] -= detections[0][0]
        detections[0][6] -= detections[0][0]
        detections[0][8] -= detections[0][0]
        detections[0][10] -= detections[0][0]
        detections[0][5] -= detections[0][1]
        detections[0][7] -= detections[0][1]
        detections[0][9] -= detections[0][1]
        detections[0][11] -= detections[0][1]
        detections[0][4:] /= ((x2-x1) / 112)
        if (x2 - x1) < 60: return # min size of 60 for now?
        cropped = orig[int(x1):int(x2), int(y1):int(y2)]
        face_img = cv2.resize(cropped, (112, 112))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
        return face_img, detections[0]

if __name__ == '__main__':
    blazeface = BlazeFace()

    img = cv2.imread("img3.jpeg")
    img, faces = img_to_face(img)
    

    # The ys are correct but not the Xs?
    facial4points = [
    [faces[4], faces[5]],   # left eye (kp1)
    [faces[6], faces[7]],   # right eye (kp0)
    [faces[8], faces[9]],   # nose (kp2)
    [faces[10], faces[11]]] # mouth (kp3) → duplicate

    cv2.imwrite("my_pic.jpg", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    print(facial4points, "RORY")

    points = [(int(pt[0]), int(pt[1])) for pt in facial4points]
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]

    for point, color in zip(points, colors): 
        cv2.circle(img, point, radius=4, color=color, thickness=-1)

    # NOW save the image with points
    cv2.imwrite("my_pic_drawn.jpg", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cv2.imwrite("my_pic.jpg", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    print(facial4points, "RORY")

    img = Image.open("img3.jpeg").convert('RGB')
    boxes, faces = mtcnn_model.align_multi(img, limit=1)
    face = faces[0]
    aligned_rgb_img = np.array(face)

    cv2.imwrite("aligned_pic.jpg", cv2.cvtColor(aligned_rgb_img, cv2.COLOR_BGR2RGB))
    

