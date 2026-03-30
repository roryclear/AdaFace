from face_alignment import align
from inference import load_pretrained_model, to_input
import cv2
import numpy as np

model = load_pretrained_model('ir_50')

img = cv2.imread('messi_aligned.jpg')
bgr_input = to_input(img)
feature, _ = model(bgr_input)
print("messi feature =", feature, feature.shape)

img = cv2.imread('messi_aligned2.jpg')
bgr_input = to_input(img)
feature2, _ = model(bgr_input)
print("messi2 feature =", feature2, feature2.shape)

path = 'ronaldo.jpg'
img = cv2.imread('ronaldo_aligned.jpg')
bgr_input = to_input(img)
feature3, _ = model(bgr_input)
print("ronaldo feature =", feature3, feature3.shape)

print("messi to messi2 =", feature @ feature2.T)
print("messi to ronaldo =", feature @ feature3.T)
print("messi2 to ronaldo", feature2 @ feature3.T)

