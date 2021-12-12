import cv2
from detect import DetectYolov5

source = 'random.jpeg'
image = cv2.imread(source)
model_path = './model_data/yolov5s.pt'

detector = DetectYolov5(model_path)

img = detector.detect_image(image)

cv2.imshow('Frame',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
