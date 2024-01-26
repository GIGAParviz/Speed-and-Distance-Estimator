import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import colors, Annotator
import numpy as np
from time import perf_counter , time
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--videopath', type=str , help="Enter youre Video path")
opt = parser.parse_args()



model = YOLO(r"C:\Users\ASUS\OneDrive\Desktop\YOLO-8\yolov8m.pt")


cap = cv2.VideoCapture(opt.videopath)


out = cv2.VideoWriter(r'C:\Users\ASUS\OneDrive\Desktop\Minecraft.1.18.2.Cracked-Par30Game\visioneye-pinpoint.avi', cv2.VideoWriter_fourcc(*'MJPG'),
                    30, (int(cap.get(3)), int(cap.get(4))))

center_point = (1000, int(cap.get(4)))

while True:
    start = perf_counter()
    ret, im0 = cap.read()
    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    s = time()

    results = model.track(im0, persist=True)
    boxes = results[0].boxes.xyxy.cpu()
    
    end = perf_counter()
    e = time()
    total = end - start
    fps = 1/total
    sec = e - s 
    
    cv2.putText(im0 ,(f"FPS: {str(int(fps))}"), (100,100) , 0 , 1 ,color=(255,255,255) ,thickness= 2, lineType=cv2.LINE_AA )
    cv2.putText(im0 ,("D is Distance and V is Velocity"), (100,150) , 0 , 1 ,color=(255,255,255) ,thickness= 2, lineType=cv2.LINE_AA )

    if results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            annotator = Annotator(im0, line_width=2)
            
            start = perf_counter()
            
            x_middle = int((box[0] + box[2])/2)
            y_middle = int((box[1] + box[3])/2)
            
            distance = np.sqrt(np.abs(((center_point[0] - (x_middle))**2) + ((center_point[1] - (y_middle)**2))))
            distance = distance / 100
            d = np.round(distance , 2)

            annotator.box_label(box,  color=colors(int(track_id)))
            annotator.visioneye(box, center_point)
                    
            v = float(distance/100) / sec   
            v = np.round(v , 2 )
            text=(f'D ~ {int(d)}m & V ~ {v}m/s')
            cv2.putText(im0 ,(text) , (int(box[0]-22) , int(box[1]-5)) , 0 , 0.5 ,color=colors(int(track_id)) ,thickness= 2, lineType=cv2.LINE_AA )

    out.write(im0)
    cv2.namedWindow("visioneye-pinpoint", cv2.WINDOW_FREERATIO)
    cv2.imshow("visioneye-pinpoint", im0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()

