import dlib
import cv2
import json
import random
from tqdm import tqdm
import os

def bbox(folder_path,video):
    detector = dlib.get_frontal_face_detector()
    path = os.path.join(folder_path, video)
    url = path.split('.mp4')[0]
    print('url',url)
    # Open the video file
    video_capture = cv2.VideoCapture(path)

    # Get the frame dimensions
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    accelerate = False

    if not accelerate:
        # Get total number of frames in the video
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        smallest_bb = None
        # Iterate over frames with tqdm to display a progress bar
        for i in tqdm(range(total_frames), desc='Processing frames', unit='frame'):
            # Read a frame from the video
            ret, frame = video_capture.read()
            if not ret:
                break  # Break the loop if no frame is captured

            if i!=total_frames-1:
                continue
            # Convert the frame to grayscale (dlib works on grayscale images)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = detector(gray_frame)

            # Find the smallest bounding box containing the face in the frame
            for face in faces:
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                x_min, y_min, x_max, y_max = x / frame_width, y / frame_height, (x + w) / frame_width, (y + h) / frame_height
                bounding_box = [x_min, y_min, x_max, y_max]
                smallest_bb = bounding_box

        # Release the video capture object
        video_capture.release()
        return smallest_bb
        

def test_bbox(folder_path, target_path):
    with open('utils/bounding_boxes.json', 'r') as json_file:
        bounding_boxes = json.load(json_file)
        for vid, bb_coordinates in bounding_boxes.items():
            path = os.path.join(folder_path, vid+'.mp4')
            # print(path)
            if path!="utils/video-clip/aTCMM7U4dNM-00:01:45.000-00:01:52.000.mp4" and path!="video-clip/_lfi80q8xQg-00:00:05.620-00:00:13.140.mp4":
                continue
            video_capture = cv2.VideoCapture(path)

            # Get the frame dimensions
            frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
            # Choose a random frame number
            random_frame = 10

            # Read and display the random frame
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, random_frame)
            ret, frame = video_capture.read()
            if ret:
                # Convert the frame to grayscale (dlib works on grayscale images)
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Convert normalized bounding box coordinates to pixel values
                x_min, y_min, x_max, y_max = bb_coordinates
                x_min, y_min, x_max, y_max = int(x_min * frame_width), int(y_min * frame_height), \
                                            int(x_max * frame_width), int(y_max * frame_height)
             
                # Draw the bounding box on the frame
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
                # Display the frame with the bounding box
                os.makedirs(target_path, exist_ok=True)
                img_name = 'bbox_'+vid+'.jpg'
                output_path = os.path.join(target_path, img_name)
                cv2.imwrite(output_path, frame)
            else:
                print("Error reading frame from the video.")
            video_capture.release()