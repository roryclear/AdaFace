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

import numpy as np

def rotate_point(point, center, angle_rad):
    """Rotate a single point around center."""
    x, y = point
    cx, cy = center
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    x_new = cos_a * (x - cx) - sin_a * (y - cy) + cx
    y_new = sin_a * (x - cx) + cos_a * (y - cy) + cx
    return np.array([x_new, y_new])

def warp_affine_np(img, matrix, output_shape):
    out_h, out_w = output_shape[:2]
    in_h, in_w = img.shape[:2]
    a, b, tx = matrix[0]
    c, d, ty = matrix[1]
    det = a * d - b * c
    inv_a = d / det
    inv_b = -b / det
    inv_c = -c / det
    inv_d = a / det
    inv_tx = (b * ty - d * tx) / det
    inv_ty = (c * tx - a * ty) / det
    y_out, x_out = np.ogrid[:out_h, :out_w]
    x_in = inv_a * x_out + inv_b * y_out + inv_tx
    y_in = inv_c * x_out + inv_d * y_out + inv_ty
    valid_mask = (x_in >= 0) & (x_in < in_w - 1) & (y_in >= 0) & (y_in < in_h - 1)
    output = np.zeros((out_h, out_w, img.shape[2]), dtype=img.dtype)
    x_in_valid = x_in[valid_mask]
    y_in_valid = y_in[valid_mask]
    x0 = np.floor(x_in_valid).astype(int)
    y0 = np.floor(y_in_valid).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1
    x0 = np.clip(x0, 0, in_w - 1)
    y0 = np.clip(y0, 0, in_h - 1)
    x1 = np.clip(x1, 0, in_w - 1)
    y1 = np.clip(y1, 0, in_h - 1)
    dx = x_in_valid - x0
    dy = y_in_valid - y0
    dx = dx[:, np.newaxis]
    dy = dy[:, np.newaxis]
    top = img[y0, x0] * (1 - dx) + img[y0, x1] * dx
    bottom = img[y1, x0] * (1 - dx) + img[y1, x1] * dx
    interpolated = top * (1 - dy) + bottom * dy
    output[valid_mask] = interpolated
    return output


def align_face_np(img, facial_points, reference_points, output_size=(112, 112)):
    h, w = img.shape[:2]
    
    # Extract eye points
    left_eye = np.array(facial_points[0], dtype=np.float32)
    right_eye = np.array(facial_points[1], dtype=np.float32)
    
    ref_left = np.array(reference_points[0], dtype=np.float32)
    ref_right = np.array(reference_points[1], dtype=np.float32)
    
    # 1. ROTATION: Make eyes horizontal
    eye_vector = right_eye - left_eye
    angle = np.arctan2(eye_vector[1], eye_vector[0])  # Radians
    
    # Rotate eye points around center of image
    center = np.array([w / 2, h / 2])
    left_eye_rot = rotate_point(left_eye, center, -angle)
    right_eye_rot = rotate_point(right_eye, center, -angle)
    
    # 2. SCALE: Match eye distance to reference
    current_eye_dist = np.linalg.norm(right_eye_rot - left_eye_rot)
    ref_eye_dist = np.linalg.norm(ref_right - ref_left)
    scale = ref_eye_dist / current_eye_dist
    
    # 3. TRANSLATION: Center on reference
    current_center = (left_eye_rot + right_eye_rot) / 2
    ref_center = (ref_left + ref_right) / 2
    
    # Build transformation matrix: scale and translate
    # First, translate to origin, scale, then translate to ref center
    # Combined: [scale, 0, tx]
    #            [0, scale, ty]
    tx = ref_center[0] - current_center[0] * scale
    ty = ref_center[1] - current_center[1] * scale
    
    # Create rotation matrix for the whole image
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    
    # Combine transformations: scale_trans @ rot
    combined_matrix = np.array([
        [scale * cos_a, -scale * sin_a, scale * (center[0] - cos_a * center[0] + sin_a * center[1]) + tx],
        [scale * sin_a, scale * cos_a, scale * (center[1] - sin_a * center[0] - cos_a * center[1]) + ty]
    ])
    img_aligned = warp_affine_np(img, combined_matrix, output_size)
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

    aligned_face = align_face_np(img, facial4points, REFERENCE_FACIAL_POINTS)

    cv2.imwrite("aligned_simple.jpg", cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB))

    print(facial4points, "RORY")

    img = Image.open("img3.jpeg").convert('RGB')
    boxes, faces = mtcnn_model.align_multi(img, limit=1)
    face = faces[0]
    aligned_rgb_img = np.array(face)

    cv2.imwrite("aligned_pic.jpg", cv2.cvtColor(aligned_rgb_img, cv2.COLOR_BGR2RGB))
    

