from ultralytics import YOLO
import cv2
import os
from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated


def runInfer():
    cap = cv2.VideoCapture("./SourceVideo3.mp4")

    output_path = './Test/output_video2.mp4'
    record = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 60, (1920, 1080))
    
    model = YOLO('C:/Users/Admin/Desktop/GoaliePositioningGuide/runs/detect/train8/weights/best.pt')
   
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        
        if success:
            # Run YOLOv8 inference on the frame
            results = model.predict(frame, verbose=False)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            # cv2.imshow("YOLOv8 Inference", annotated_frame)
          
            #save the frame
            record.write(annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    record.release() #Stop Recording Video if saveVid is True
    cv2.destroyAllWindows()


if __name__ == "__main__":
    runInfer()
