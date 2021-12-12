import cv2
import time
from detector import DetectYolov5
from utils.videocapture import CustomVideoCapture


model_path = './model_data/yolov5l.pt'
detector = DetectYolov5(model_path)

url = 'rtsp://admin:AiBi@8899@192.168.101.80:554'
cap = CustomVideoCapture(url)

while True:
    t1 = time.time()
    frame = cap.read()
    image, bboxes, names = detector.detect_image(frame, conf_threshold = 0.6, resize_factor = 0.4)
    #image = cv2.resize(img,(640,640))
    t2 = time.time()
    print('Processing Time : {} s'.format(round((t2-t1),3)))

    cv2.imshow('IP Camera Object Detection',image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  

cv2.destroyAllWindows()
