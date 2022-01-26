import cv2
import time
from detector import DetectYolov5
from utils.videocapture import CustomVideoCapture


model_path = './model_data/yolov5l.pt'
detector = DetectYolov5(model_path)
url = 0          # use Url for RTSP Stream 

cap = CustomVideoCapture(url)

while True:
    t1 = time.time()
    frame = cap.read()
    image, bboxes, names = detector.detect_image(frame, conf_threshold = 0.2, resize_factor = 0.7)
    image = cv2.resize(image,(640,640))
    t2 = time.time()
    print('Processing Time : {} s'.format(round((t2-t1),3)))

    cv2.imshow('Webcam Object Detection',image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  

cv2.destroyAllWindows()
