import cv2
from detector import DetectYolov5
import time

source = 'random.jpeg'
#source = '0.jpg'
image = cv2.imread(source)

model_path = './model_data/yolov5m.pt'
#model_path = 'C:/Users/User/Desktop/ACI/yolov5_models/crowd/crowdhuman_yolov5m_deepakcrk.pt'
detector = DetectYolov5(model_path)

t1 = time.time()
img, bboxes, names = detector.detect_image(image, conf_threshold = 0.2)
t2 = time.time()
print('Processing Time : {}'.format(round(t2-t1,3)))

cv2.imshow('Frame',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
