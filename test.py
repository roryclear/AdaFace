from face_alignment import align
from inference import load_pretrained_model, to_input

model = load_pretrained_model('ir_50')
path = 'messi.jpg'
aligned_rgb_img = align.get_aligned_face(path)
bgr_input = to_input(aligned_rgb_img)
feature, _ = model(bgr_input)
print("rory feature =",feature, feature.shape)

path = 'messi2.avif'
aligned_rgb_img = align.get_aligned_face(path)
bgr_input = to_input(aligned_rgb_img)
feature2, _ = model(bgr_input)
print("rory feature =",feature, feature.shape)

path = 'ronaldo.jpg'
aligned_rgb_img = align.get_aligned_face(path)
bgr_input = to_input(aligned_rgb_img)
feature3, _ = model(bgr_input)
print("rory feature =",feature, feature.shape)

print("messi to messi2 =", feature @ feature2.T)
print("messi to ronaldo =", feature @ feature3.T)
print("messi2 to ronaldo", feature2 @ feature3.T)

