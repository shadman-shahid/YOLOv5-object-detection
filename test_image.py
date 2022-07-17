import cv2
from detector import DetectYolov5
import time

source = 'random.jpeg'
image = cv2.imread(source)

model_path = './models/yolov5m.pt'
# model_path = './model_data/crowdhuman_yolov5m.pt'

detector = DetectYolov5(model_path)

t1 = time.time()
img, bboxes, names = detector.detect_image(image, conf_threshold=0.2)
t2 = time.time()
print('Processing Time : {}'.format(round(t2-t1,3)))

cv2.imshow('Frame',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
