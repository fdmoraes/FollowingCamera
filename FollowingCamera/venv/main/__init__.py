import face_recognition
import cv2
import numpy as np
import platform
#from simple_pid import PID

def running_on_jetson_nano():
    # To make the same code work on a laptop or on a Jetson Nano, we'll detect when we are running on the Nano
    # so that we can access the camera correctly in that case.
    # On a normal Intel laptop, platform.machine() will be "x86_64" instead of "aarch64"
    return platform.machine() == "aarch64"

def get_jetson_gstreamer_source(capture_width=1280, capture_height=720, display_width=1280, display_height=720, framerate=60, flip_method=0):
    """
    Return an OpenCV-compatible video source description that uses gstreamer to capture video from the camera on a Jetson Nano
    """
    return (
            'nvarguscamerasrc ! video/x-raw(memory:NVMM), ' +
            'width=(int){capture_width}, height=(int){capture_height}, ' +
            'format=(string)NV12, framerate=(fraction){framerate}/1 ! ' +
            'nvvidconv flip-method={flip_method} ! ' +
            'video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! ' +
            'videoconvert ! video/x-raw, format=(string)BGR ! appsink'
            )

def get_most_similar_face(faces, base_face):
    face_distances = face_recognition.face_distance(faces, base_face)
    best_match_index = np.argmin(face_distances)
    if face_distances[best_match_index] < 0.65:
        return best_match_index
    else:
        return None

def draw_rectangle_on_faces(face_locations, better_face, frame):

    height = frame.shape[0]
    width = frame.shape[1]
    #channels = frame.shape[2]

    y1 = 0
    y2 = 0
    x1 = 0
    x2 = 0
    initialized = False

    for face in face_locations:
        top, right, bottom, left = face
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        if better_face == face:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            y1 = top
            y2 = height - bottom
            x1 = left
            x2 = width - right
            initialized = True
        elif better_face is None:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        else:
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 255), 2)

    if initialized == True:
        return x1 - x2, y1 - y2
    else:
        return None, None

def main_loop():
    # Get access to the webcam. The method is different depending on if this is running on a laptop or a Jetson Nano.
    if running_on_jetson_nano():
        # Accessing the camera with OpenCV on a Jetson Nano requires gstreamer with a custom gstreamer source string
        video_capture = cv2.VideoCapture(get_jetson_gstreamer_source(), cv2.CAP_GSTREAMER)
    else:
        # Accessing the camera with OpenCV on a laptop just requires passing in the number of the webcam (usually 0)
        # Note: You can pass in a filename instead if you want to process a video file instead of a live camera stream
        video_capture = cv2.VideoCapture(0)

    state = 0
    last_face_encodings = []

#    pidX = PID(1, 0.1, 0.05, setpoint=0)
#    pidY = PID(1, 0.1, 0.05, setpoint=0)

#    pidX.sample_time = 0.01
#    pidY.sample_time = 0.01

    #pidX.output_limits = (0, 100)
    #pidY.output_limits = (0, 100)

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Find all the face locations and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        if state == 0:

            if len(face_locations) != 1:
                last_face_encodings = []
            else:
                if len(last_face_encodings) == 0:
                    last_face_encodings.append(face_encodings[0])
                elif len(last_face_encodings) < 10:
                    current_face_encoding = []
                    current_face_encoding.append(last_face_encodings[-1])

                    best_match_index = get_most_similar_face(current_face_encoding, face_encodings[0])
                    if best_match_index is None:
                        last_face_encodings = []
                    else:
                        last_face_encodings.append(face_encodings[0])
                else:
                    current_face_encoding = []
                    current_face_encoding.append(last_face_encodings[-1])

                    best_match_index = get_most_similar_face(current_face_encoding, face_encodings[0])
                    if best_match_index is None:
                        last_face_encodings = []
                    else:
                        last_face_encodings.append(face_encodings[0])
                        state = 1

                draw_rectangle_on_faces(face_locations, None, frame)

        else:
            if len(face_locations) == 0:
                last_face_encodings = []
                state = 0
            else:
                best_match_index = get_most_similar_face(current_face_encoding, face_encodings[0])
                if best_match_index is None:
                    draw_rectangle_on_faces(face_locations, None, frame)
                else:
                    better_face = face_locations[best_match_index]

                    x, y = draw_rectangle_on_faces(face_locations, better_face, frame)
#                    outputX = pidX(x)
#                    outputY = pidY(y)
                    last_face_encodings[-1] = face_encodings[best_match_index]

#                    if not (x is None and y is None):
#                        print("%d -> %d || %d -> %d" % (x, outputX, y, outputY))

        # Display the final frame of video with boxes drawn around each detected fames
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_loop()