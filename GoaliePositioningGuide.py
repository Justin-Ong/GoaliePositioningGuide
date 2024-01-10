import ctypes
import cv2
import dxcam
import numpy as np
import tkinter as tk
import multiprocessing

from ultralytics import YOLO


# Detects display resolution
def get_resolution():
    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    width = user32.GetSystemMetrics(0)
    height = user32.GetSystemMetrics(1)
    return (width, height)


# Runs the YOLOv8 model to detect the position of the core
def find_image_on_screen(conn):
    # Load custom YOLOv8 model
    model = YOLO('./best.pt')
    
    # Set confidence threshold for sending detected object's position to the drawing process
    threshold=0.8
    
    # Create and start screen capture thread
    # Setting output_color='BGR' to record the screenshots as RGB
    # Otherwise, screenshots are in BGR
    # I don't know why it works this way
    camera = dxcam.create(device_idx=0, output_idx=0, output_color='BGR')
    camera.start(target_fps=144)

    while True:
        screenshot = camera.get_latest_frame()
        results = model(screenshot, stream=True, verbose=False)

        for result in results:
            for box in result.boxes:
                # Need to copy result tensor from GPU to CPU and convert to nparray for processing
                data = box.cpu().numpy()
                coords = data.xyxy
                minx, miny, maxx, maxy = coords[0, 0], coords[0, 1], coords[0, 2], coords[0, 3]
                if data.conf > threshold:
                    conn.send(((minx + maxx) / 2, (miny + maxy) / 2))


# Creates a transparent overlay window and draws a red line from the centre of the goal to the core
def draw_red_line(conn):
    root = tk.Tk()
    root.overrideredirect(True)
    root.lift()
    root.wm_attributes('-topmost', True)
    root.wm_attributes('-disabled', True)
    root.wm_attributes('-transparentcolor', 'white')

    width, height = get_resolution()
    canvas = tk.Canvas(root, bg='white', height=height, width=width)
    canvas.pack()

    initialised = False
    
    while True:
        end_point = conn.recv()
        if initialised:
            canvas.coords(line, 215, 540, end_point[0], end_point[1])
        else:
            line = canvas.create_line(215, 540, end_point[0], end_point[1], fill='red', width=1)
            initialised = True
        canvas.update()


if __name__ == '__main__':
    conn1, conn2 = multiprocessing.Pipe()
    image_rec_process = multiprocessing.Process(target=find_image_on_screen, args=(conn1,))
    line_drawing_process = multiprocessing.Process(target=draw_red_line, args=(conn2,))
    image_rec_process.daemon = True
    line_drawing_process.daemon = True
    image_rec_process.start()
    line_drawing_process.start()
    input('Press ENTER to stop.')
