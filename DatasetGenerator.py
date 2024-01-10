import cv2
import os


def generate_dataset():
    cap = cv2.VideoCapture("./SourceVideo5.mp4")
    i = 0
    frame_skip = 10
    frame_count = 9336
    
    training_data_path = "./datasets/Dataset/TrainingImages"
    validation_data_path = "./datasets/Dataset/ValidationImages"
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if i > frame_skip - 1:
            frame_count += 1
            # if (frame_count % 2):
                # img_name = os.path.join(training_data_path, str(frame_count) + ".jpg")
            # else:
                # img_name = os.path.join(validation_data_path, str(frame_count) + ".jpg")
            img_name = os.path.join(validation_data_path, str(frame_count) + ".jpg")
            cv2.imwrite(img_name, frame)
            i = 0
            continue
        i += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    generate_dataset()
