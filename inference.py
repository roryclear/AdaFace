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
    [73.53179932, 51.69630051],
    [56, 71.73660278],
    [56, 92.3655014 ]
]

def align_face_simple(img, facial_points, reference_points):
    """
    Simple alignment: rotate to make eyes horizontal, then scale and translate
    to match reference points.
    """
    # Extract points
    left_eye = np.array(facial_points[0])
    right_eye = np.array(facial_points[1])
    
    # 1. ROTATION: Make eyes horizontal
    # Calculate angle between eyes
    eye_vector = right_eye - left_eye
    angle = np.degrees(np.arctan2(eye_vector[1], eye_vector[0]))
    
    # Rotate image around center
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    img_rotated = cv2.warpAffine(img, rotation_matrix, (w, h))
    
    # Rotate all facial points
    points_rotated = []
    for point in facial_points:
        pt = np.array([point[0], point[1], 1])
        new_pt = rotation_matrix @ pt
        points_rotated.append(new_pt)
    
    # 2. SCALE: Match the eye distance to reference
    current_eye_dist = np.linalg.norm(points_rotated[1] - points_rotated[0])
    ref_eye_dist = np.linalg.norm(np.array(reference_points[1]) - np.array(reference_points[0]))
    scale = ref_eye_dist / current_eye_dist
    
    # 3. TRANSLATION: Center on reference
    current_center = np.mean(points_rotated[:2], axis=0)  # center of eyes
    ref_center = np.mean(reference_points[:2], axis=0)    # center of reference eyes
    
    # Apply scale and translation in one matrix
    h, w = img_rotated.shape[:2]
    center_pt = (w // 2, h // 2)
    
    # Create transformation matrix: scale then translate
    tx = ref_center[0] - (current_center[0] * scale)
    ty = ref_center[1] - (current_center[1] * scale)
    
    transform_matrix = np.array([
        [scale, 0, tx],
        [0, scale, ty]
    ], dtype=np.float32)
    
    # Apply transformation
    output_size = (112, 112)  # Standard face size
    img_aligned = cv2.warpAffine(img_rotated, transform_matrix, output_size)
    return img_aligned

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



    aligned_face = align_face_simple(img, facial4points, REFERENCE_FACIAL_POINTS)

    cv2.imwrite("aligned_simple.jpg", cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB))

    print(facial4points, "RORY")

    img = Image.open("img3.jpeg").convert('RGB')
    boxes, faces = mtcnn_model.align_multi(img, limit=1)
    face = faces[0]
    aligned_rgb_img = np.array(face)

    cv2.imwrite("aligned_pic.jpg", cv2.cvtColor(aligned_rgb_img, cv2.COLOR_BGR2RGB))
    

