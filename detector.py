import cv2
import math
from pathlib import Path
import time
from PIL import Image, ImageDraw

from google.colab.patches import cv2_imshow
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np


MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green


def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
          landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
          annotated_image,
          hand_landmarks_proto,
          solutions.hands.HAND_CONNECTIONS,
          solutions.drawing_styles.get_default_hand_landmarks_style(),
          solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image


def update_color(image, color):
    # Convert image to RGB (OpenCV uses BGR by default)
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = np.array(image)

    # Define the white color in RGB
    white = 255

    # Create a mask for non-white pixels
    mask = np.logical_not(np.all(image_rgb == white, axis=-1))

    # Create an output image with the same size and initialize with white color
    output_image = np.full_like(image_rgb, white)  # White in RGB

    # Set color in the output image where mask is True
    output_image[mask] = color

    # Convert back to BGR for OpenCV compatibility
    # output_image_bgr = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

    # Save or display the result
    # cv2.imwrite('output_image.jpg', output_image_bgr)
    # cv2.imshow('Result', output_image_bgr)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # return output_image_bgr
    return Image.fromarray(output_image, mode='RGB')


def remove_files_with_pattern(directory, pattern):
    # Convert the directory to a Path object
    dir_path = Path(directory)

    # List all files matching the pattern
    files_to_remove = dir_path.glob(pattern)

    # Remove each file
    for file_path in files_to_remove:
        try:
            file_path.unlink()  # Remove the file
            print(f"Removed: {file_path}")
        except Exception as e:
            print(f"Error removing {file_path}: {e}")


def get_file_name(teacher_or_stu,
                  input_or_output,
                  rotate_or_updown,
                  suffix,
                  index_fomat=None,
                  is_print=False):
    assert teacher_or_stu in ['teacher', 'stu'], teacher_or_stu
    assert input_or_output in ['input', 'output'], input_or_output
    assert rotate_or_updown in ['rotate', 'updown'], rotate_or_updown
    assert suffix in ['.mp4', '.png'], suffix

    if suffix in ['.png']:
        assert index_fomat in ['%04d', '*'], index_fomat

    file_name_parts = [teacher_or_stu,
                       input_or_output,
                       rotate_or_updown]

    if suffix in ['.png']:
        file_name_parts.append(index_fomat)

    file_name = '_'.join(file_name_parts)
    file_full_name = rf'{file_name}{suffix}'

    if is_print:
        print(file_full_name)

    return file_full_name


def concat_images(left_image, right_image):
    right_image.resize((left_image.width, left_image.height))
    # Get dimensions of both images
    left_width, left_height = left_image.size
    right_width, right_height = right_image.size

    # Create a new image with a width that is the sum of both images' widths
    # and height that is the maximum of both images' heights
    new_width = left_width + right_width
    new_height = max(left_height, right_height)

    # Create a new blank image with the calculated dimensions
    new_image = Image.new('RGB', (new_width, new_height))

    # Paste the left image at the (0, 0) position
    new_image.paste(left_image, (0, 0))

    # Paste the right image at the (left_width, 0) position
    new_image.paste(right_image, (left_width, 0))

    return new_image


def concat_images_list(rotate_or_updown):
    image_paths_list = []
    dir_path = Path('./frames')

    for teacher_or_stu in ['teacher', 'stu']:
        file_name_pattern = get_file_name(teacher_or_stu,
                                                 'output',
                                                 rotate_or_updown,
                                                 '.png',
                                                 index_fomat='*')
        image_paths = list(dir_path.glob(file_name_pattern))

        image_paths.sort()

        image_paths_list.append(image_paths)

    for left_image_path, right_image_path in zip(*image_paths_list):
        left_image = Image.open(str(left_image_path))
        right_image = Image.open(str(right_image_path))
        new_image = concat_images(left_image, right_image)

        left_index = len(left_image_path.name) - 1
        right_index = len(right_image_path.name) - 1

        while (left_index >= 0 and right_index >= 0 and
               left_image_path.name[left_index] == right_image_path.name[right_index]):
            left_index -= 1
            right_index -= 1

        output_file_name = left_image_path.name[left_index + 1:]

        if output_file_name.startswith('_'):
            output_file_name = output_file_name[len('_'):]

        output_file_path = left_image_path.parent / output_file_name
        print(output_file_path)
        new_image.save(str(output_file_path))

    print('Completed!')


class VideoParser:
    def __init__(self,
                 straight_arrow_path,
                 clockwise_arrow_path,
                 running_mode):
        self.frame_prefix_dict = {
            'teacher_clockwise': 'frame_teacher_clockwise',
            'teacher_up_down': 'frame_teacher_up_down',
            'student_clockwise': 'frame_student_clockwise',
            'student_up_down': 'frame_student_up_down',
        }
        self.clockwise_arrow_path = clockwise_arrow_path
        self.straight_arrow_path = straight_arrow_path
        # self.input_video_path = input_video_path
        # self.recognizer = self.load_recognizer(running_mode)
        self.detector = self.load_detector(running_mode)

        self.up_image = Image.open(self.straight_arrow_path)
        self.down_image = self.up_image.transpose(Image.FLIP_TOP_BOTTOM)

        self.clockwise_image = Image.open(self.clockwise_arrow_path)
        self.counter_clockwise_image = self.clockwise_image.transpose(Image.FLIP_LEFT_RIGHT)

        red_color = [255, 0, 0]

        self.red_up_image = update_color(self.up_image, red_color)
        self.red_down_image = self.red_up_image.transpose(Image.FLIP_TOP_BOTTOM)

        self.red_clockwise_image = update_color(self.clockwise_image, red_color)
        self.red_counter_clockwise_image = self.red_clockwise_image.transpose(Image.FLIP_LEFT_RIGHT)

    def get_frames(self, input_video_path, seconds):
        frame_infos = []

        # Start recording time
        start_time = time.time()

        # Load the video using OpenCV’s VideoCapture.
        cap = cv2.VideoCapture(input_video_path)

        # Obtain the frame rate of the video.
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Get the height of the frames in the video
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Get the width of the frames in the video
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print('fps:', fps,
              'frame_height:', frame_height,
              'frame_width:', frame_width)

        # Calculate the duration of each frame in milliseconds.
        frame_duration_ms = 1000 / fps

        # Initialize the frame timestamp.
        frame_timestamp_ms = 0

        frame_index = 0

        # Loop through each frame in the video using VideoCapture#read().
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Calculate the timestamp for the current frame.
            frame_timestamp_ms = frame_index * frame_duration_ms

            second = int(frame_timestamp_ms / 1000)

            frame_info = {
                'frame_index': frame_index,
                'frame_timestamp_ms': frame_timestamp_ms,
                'second': second,
                'frame': frame
            }

            # Increment the frame index.
            frame_index += 1

            if second not in seconds:
                continue

            frame_infos.append(frame_info)

        # Release resources.
        cap.release()
        cv2.destroyAllWindows()

        # End time
        end_time = time.time()

        # Calculate the duration
        total_seconds = end_time - start_time

        # Convert to minutes and seconds
        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60

        # Print the time duration
        print(f"Time duration: {minutes} minutes and {seconds:.2f} seconds")

        return frame_infos

    def save_frames(self, frame_infos):
        for frame_info in frame_infos:
            file_name = r'{}_{}.png'.format(frame_info['second'],
                                         frame_info['frame_index'])
            cv2.imwrite(file_name, frame_info['frame'])

    def load_recognizer(self, running_mode):
        base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
        options = vision.GestureRecognizerOptions(base_options=base_options,
                                                  running_mode=running_mode,
                                                  num_hands=2,
                                                  min_hand_detection_confidence=0.1,
                                                  min_hand_presence_confidence=0.1,
                                                  min_tracking_confidence=0.1)
        recognizer = vision.GestureRecognizer.create_from_options(options)
        return recognizer

    def load_detector(self, running_mode):
        hand_base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        hand_options = vision.HandLandmarkerOptions(base_options=hand_base_options,
                                                    running_mode=running_mode,
                                                    num_hands=2,
                                                    min_hand_detection_confidence=0.1,
                                                    min_hand_presence_confidence=0.1,
                                                    min_tracking_confidence=0.1)
        detector = vision.HandLandmarker.create_from_options(hand_options)

        return detector

    def resize(self, image, dest_height, dest_width):
        h, w = image.shape[:2]

        if h < w:
            img = cv2.resize(image, (dest_width, math.floor(h/(w/dest_width))))
        else:
            img = cv2.resize(image, (math.floor(w/(h/dest_height)), dest_height))

        return img

    def detect(self, image_file_name, dest_shape):
        input_image = cv2.imread(image_file_name)
        image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        dest_height, dest_width = dest_shape
        image = self.resize(image, dest_height, dest_width)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        recognition_result = self.recognizer.recognize(image)
        if len(recognition_result.handedness) != 2:
          print(image_file_name, ', '.join(handedness[0].display_name for handedness in recognition_result.handedness))
          annotated_image = draw_landmarks_on_image(image.numpy_view(), recognition_result)
          cv2_imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))


        detection_result = self.detector.detect(image)

        if len(detection_result.handedness) != 2:
            print(image_file_name)
            print(', '.join('{}: {:.3}'.format(handedness[0].display_name, handedness[0].score) for handedness in detection_result.handedness))
            annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
            cv2_imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    def process_clockwise_video(self, input_video_path, output_video_path, is_teacher):
        # Start recording time
        start_time = time.time()
        dir_path = Path('./frames')
        dir_path.mkdir(parents=True, exist_ok=True)
        frame_prefix_key = 'teacher_clockwise' if is_teacher else 'student_clockwise'
        frame_prefix = self.frame_prefix_dict[frame_prefix_key]
        remove_files_with_pattern(dir_path, rf'{frame_prefix}_*.png')
        annotated_images = []

        # Load the video using OpenCV’s VideoCapture.
        cap = cv2.VideoCapture(input_video_path)

        # Obtain the frame rate of the video.
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Get the height of the frames in the video
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Get the width of the frames in the video
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print('fps:', fps,
              'frame_height:', frame_height,
              'frame_width:', frame_width)

        # Calculate the duration of each frame in milliseconds.
        frame_duration_ms = 1000 / fps

        # Initialize the frame timestamp.
        frame_timestamp_ms = 0

        frame_index = 0

        # Loop through each frame in the video using VideoCapture#read().
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Calculate the timestamp for the current frame.
            frame_timestamp_ms = frame_index * frame_duration_ms

            second = int(frame_timestamp_ms / 1000)

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotated_image = Image.fromarray(image, mode='RGB')

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            detection_result = self.detector.detect_for_video(mp_image, int(frame_timestamp_ms))

            handedness_list = detection_result.handedness
            hand_landmarks_list = detection_result.hand_landmarks

            print(frame_index,
                  ', '.join('{}: {:.3}'.format(handedness[0].category_name,
                                               handedness[0].score)
                  for handedness in detection_result.handedness))

            if len(handedness_list) == 1:
                for handedness, hand_landmarks in zip(handedness_list, hand_landmarks_list):
                    category_name = handedness[0].category_name

                    if category_name == 'Right':
                        handedness[0].category_name = 'Left'

            if len(handedness_list) == 2:
                category_names = []
                mean_xs = []

                for handedness, hand_landmarks in zip(handedness_list, hand_landmarks_list):
                    category_name = handedness[0].category_name
                    mean_x = np.mean([landmark.x for landmark in hand_landmarks])

                    category_names.append(category_name)
                    mean_xs.append(mean_x)

                if len(set(category_names)) == 1:
                    if mean_xs[0] < mean_xs[1]:
                        left_hand_index = 0
                        right_hand_index = 1
                    else:
                        left_hand_index = 1
                        right_hand_index = 0

                    handedness_list[left_hand_index][0].category_name = 'Left'
                    handedness_list[right_hand_index][0].category_name = 'Right'

                # cv2_imshow(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            for handedness, hand_landmarks in zip(handedness_list, hand_landmarks_list):
                category_name = handedness[0].category_name

                if is_teacher:
                    if category_name == 'Left':
                        foreground_image = self.counter_clockwise_image
                    elif category_name == 'Right':
                        foreground_image = self.clockwise_image
                    else:
                        print('Error: Unkown category_name:', category_name)
                else:
                    if category_name == 'Left':
                        foreground_image = self.red_clockwise_image
                    elif category_name == 'Right':
                        foreground_image = self.red_counter_clockwise_image
                    else:
                        print('Error: Unkown category_name:', category_name)

                min_x = np.min([landmark.x for landmark in hand_landmarks]) * image.shape[1]
                max_x = np.max([landmark.x for landmark in hand_landmarks]) * image.shape[1]
                min_y = np.min([landmark.y for landmark in hand_landmarks]) * image.shape[0]

                dest_width = int(max_x + 1 - min_x)
                dest_height = int(foreground_image.height * (dest_width / foreground_image.width))

                resized_foreground_image = foreground_image.resize((dest_width, dest_height))
                foreground_arr = np.asarray(resized_foreground_image)

                bbox = [int(min_x), 0]
                # mask = np.zeros_like(foreground_arr)
                # mask[foreground_arr != 255] = 255
                mask = np.logical_not(np.all(foreground_arr == 255, axis=-1))
                mask = Image.fromarray(mask).convert('L')
                annotated_image.paste(resized_foreground_image, bbox, mask=mask)

            # annotated_image_arr = np.asarray(annotated_image)
            # annotated_image_arr = cv2.cvtColor(annotated_image_arr,
            #                                    cv2.COLOR_RGB2BGR)

            frame_file_path = dir_path / rf'{frame_prefix}_{frame_index:04d}.png'
            # cv2.imwrite(str(frame_file_path), annotated_image_arr)
            # annotated_images.append(annotated_image_arr)
            annotated_image.save(str(frame_file_path))
            annotated_images.append(annotated_image)

            # Increment the frame index.
            frame_index += 1

        # Release resources.
        cap.release()
        cv2.destroyAllWindows()

        # self.create_video_from_frames(annotated_images,
        #                               output_video_path,
        #                               fps,
        #                               codec='avc1')

        # End time
        end_time = time.time()

        # Calculate the duration
        total_seconds = end_time - start_time

        # Convert to minutes and seconds
        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60

        # Print the time duration
        print(f"Time duration: {minutes} minutes and {seconds:.2f} seconds")

        return annotated_images

    def process_images(self, teacher_or_stu, rotate_or_updown):
        # Start recording time
        start_time = time.time()

        annotated_images = []

        dir_path = Path('./frames')

        input_file_name_pattern = get_file_name(teacher_or_stu,
                                                'input',
                                                rotate_or_updown,
                                                '.png',
                                                index_fomat='*')


        image_paths = list(dir_path.glob(input_file_name_pattern))

        image_paths.sort()

        for image_path in image_paths:
            mp_image = mp.Image.create_from_file(str(image_path))
            detection_result = self.detector.detect(mp_image)

            handedness_list = detection_result.handedness
            hand_landmarks_list = detection_result.hand_landmarks

            print(image_path.name,
                  ', '.join('{}: {:.3}'.format(handedness[0].category_name,
                                               handedness[0].score)
                  for handedness in handedness_list))

            if len(handedness_list) == 1:
                for handedness, hand_landmarks in zip(handedness_list, hand_landmarks_list):
                    category_name = handedness[0].category_name

                    if rotate_or_updown in ['rotate']:
                        if category_name == 'Right':
                            handedness[0].category_name = 'Left'

                    if rotate_or_updown in ['updown']:
                        if category_name == 'Left':
                            handedness[0].category_name = 'Right'

            if len(handedness_list) == 2:
                category_names = []
                mean_xs = []

                for handedness, hand_landmarks in zip(handedness_list, hand_landmarks_list):
                    category_name = handedness[0].category_name
                    mean_x = np.mean([landmark.x for landmark in hand_landmarks])

                    category_names.append(category_name)
                    mean_xs.append(mean_x)

                if len(set(category_names)) == 1:
                    if mean_xs[0] < mean_xs[1]:
                        left_hand_index = 0
                        right_hand_index = 1
                    else:
                        left_hand_index = 1
                        right_hand_index = 0

                    handedness_list[left_hand_index][0].category_name = 'Left'
                    handedness_list[right_hand_index][0].category_name = 'Right'

            annotated_image = Image.open(str(image_path))

            for handedness, hand_landmarks in zip(handedness_list, hand_landmarks_list):
                category_name = handedness[0].category_name

                if teacher_or_stu in ['teacher']:
                    if rotate_or_updown in ['rotate']:
                        if category_name == 'Left':
                            foreground_image = self.clockwise_image
                        elif category_name == 'Right':
                            foreground_image = self.clockwise_image
                        else:
                            print('Error: Unkown category_name:', category_name)
                    elif rotate_or_updown in ['updown']:
                        if category_name == 'Left':
                            foreground_image = self.down_image
                        elif category_name == 'Right':
                            foreground_image = self.down_image
                        else:
                            print('Error: Unkown category_name:', category_name)
                elif teacher_or_stu in ['stu']:
                    if rotate_or_updown in ['rotate']:
                        if category_name == 'Left':
                            foreground_image = self.red_counter_clockwise_image
                        elif category_name == 'Right':
                            foreground_image = self.red_counter_clockwise_image
                        else:
                            print('Error: Unkown category_name:', category_name)
                    elif rotate_or_updown in ['updown']:
                        if category_name == 'Left':
                            foreground_image = self.red_up_image
                        elif category_name == 'Right':
                            foreground_image = self.red_up_image
                        else:
                            print('Error: Unkown category_name:', category_name)
                else:
                    print('Error: Unkown teacher_or_stu:', teacher_or_stu)

                min_x = np.min([landmark.x for landmark in hand_landmarks]) * annotated_image.width
                max_x = np.max([landmark.x for landmark in hand_landmarks]) * annotated_image.width

                max_dest_width = min(int(max_x + 1 - min_x), annotated_image.width // 5)
                max_dest_height = annotated_image.height // 5
                ratio = min(max_dest_width / foreground_image.width,
                            max_dest_height / foreground_image.height)
                dest_width = int(foreground_image.width * ratio)
                dest_height = int(foreground_image.height * ratio)

                resized_foreground_image = foreground_image.resize((dest_width, dest_height))
                foreground_arr = np.asarray(resized_foreground_image)

                bbox = [int(min_x), 0]
                # mask = np.zeros_like(foreground_arr)
                # mask[foreground_arr != 255] = 255
                mask = np.logical_not(np.all(foreground_arr == 255, axis=-1))
                mask = Image.fromarray(mask).convert('L')
                annotated_image.paste(resized_foreground_image, bbox, mask=mask)

            # annotated_image_arr = np.asarray(annotated_image)
            # annotated_image_arr = cv2.cvtColor(annotated_image_arr,
            #                                    cv2.COLOR_RGB2BGR)

            frame_file_path = dir_path / image_path.name.replace('input', 'output')
            # cv2.imwrite(str(frame_file_path), annotated_image_arr)
            # annotated_images.append(annotated_image_arr)
            annotated_image.save(str(frame_file_path))
            annotated_images.append(annotated_image)

        # End time
        end_time = time.time()

        # Calculate the duration
        total_seconds = end_time - start_time

        # Convert to minutes and seconds
        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60

        # Print the time duration
        print(f"Time duration: {minutes} minutes and {seconds:.2f} seconds")

        return annotated_images

    def process_up_down_video(self, input_video_path, output_video_path, is_teacher):
        # Start recording time
        start_time = time.time()
        dir_path = Path('./frames')
        dir_path.mkdir(parents=True, exist_ok=True)
        frame_prefix_key = 'teacher_up_down' if is_teacher else 'student_up_down'
        frame_prefix = self.frame_prefix_dict[frame_prefix_key]
        remove_files_with_pattern(dir_path, rf'{frame_prefix}_*.png')
        annotated_images = []

        # Load the video using OpenCV’s VideoCapture.
        cap = cv2.VideoCapture(input_video_path)

        # Obtain the frame rate of the video.
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Get the height of the frames in the video
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Get the width of the frames in the video
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print('fps:', fps,
              'frame_height:', frame_height,
              'frame_width:', frame_width)

        # Calculate the duration of each frame in milliseconds.
        frame_duration_ms = 1000 / fps

        # Initialize the frame timestamp.
        frame_timestamp_ms = 0

        frame_index = 0

        # Loop through each frame in the video using VideoCapture#read().
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Calculate the timestamp for the current frame.
            frame_timestamp_ms = frame_index * frame_duration_ms

            second = int(frame_timestamp_ms / 1000)

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotated_image = Image.fromarray(image, mode='RGB')

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            detection_result = self.detector.detect_for_video(mp_image, int(frame_timestamp_ms))

            handedness_list = detection_result.handedness
            hand_landmarks_list = detection_result.hand_landmarks

            print(frame_index,
                  ', '.join('{}: {:.3}'.format(handedness[0].category_name,
                                               handedness[0].score)
                  for handedness in detection_result.handedness))

            if len(handedness_list) == 1:
                for handedness, hand_landmarks in zip(handedness_list, hand_landmarks_list):
                    category_name = handedness[0].category_name

                    if category_name == 'Left':
                        handedness[0].category_name = 'Right'

            if len(handedness_list) == 2:
                category_names = []
                mean_xs = []

                for handedness, hand_landmarks in zip(handedness_list, hand_landmarks_list):
                    category_name = handedness[0].category_name
                    mean_x = np.mean([landmark.x for landmark in hand_landmarks])

                    category_names.append(category_name)
                    mean_xs.append(mean_x)

                if len(set(category_names)) == 1:
                    if mean_xs[0] < mean_xs[1]:
                        left_hand_index = 0
                        right_hand_index = 1
                    else:
                        left_hand_index = 1
                        right_hand_index = 0

                    handedness_list[left_hand_index][0].category_name = 'Left'
                    handedness_list[right_hand_index][0].category_name = 'Right'

                # cv2_imshow(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            for handedness, hand_landmarks in zip(handedness_list, hand_landmarks_list):
                category_name = handedness[0].category_name

                if is_teacher:
                    if category_name == 'Left':
                        foreground_image = self.down_image
                    elif category_name == 'Right':
                        foreground_image = self.down_image
                    else:
                        print('Error: Unkown category_name:', category_name)
                else:
                    if category_name == 'Left':
                        foreground_image = self.red_up_image
                    elif category_name == 'Right':
                        foreground_image = self.red_up_image
                    else:
                        print('Error: Unkown category_name:', category_name)

                min_x = np.min([landmark.x for landmark in hand_landmarks]) * image.shape[1]
                max_x = np.max([landmark.x for landmark in hand_landmarks]) * image.shape[1]
                min_y = np.min([landmark.y for landmark in hand_landmarks]) * image.shape[0]

                dest_width = int(max_x + 1 - min_x)
                dest_height = int(foreground_image.height * (dest_width / foreground_image.width))

                resized_foreground_image = foreground_image.resize((dest_width, dest_height))
                foreground_arr = np.asarray(resized_foreground_image)

                bbox = [int(min_x), 0]
                # mask = np.zeros_like(foreground_arr)
                # mask[foreground_arr != 255] = 255
                mask = np.logical_not(np.all(foreground_arr == 255, axis=-1))
                mask = Image.fromarray(mask).convert('L')
                annotated_image.paste(resized_foreground_image, bbox, mask=mask)

            # annotated_image_arr = np.asarray(annotated_image)
            # annotated_image_arr = cv2.cvtColor(annotated_image_arr,
            #                                    cv2.COLOR_RGB2BGR)

            frame_file_path = dir_path / rf'{frame_prefix}_{frame_index:04d}.png'
            # cv2.imwrite(str(frame_file_path), annotated_image_arr)
            # annotated_images.append(annotated_image_arr)
            annotated_image.save(str(frame_file_path))
            annotated_images.append(annotated_image)

            # Increment the frame index.
            frame_index += 1

        # Release resources.
        cap.release()
        cv2.destroyAllWindows()

        # self.create_video_from_frames(annotated_images,
        #                               output_video_path,
        #                               fps,
        #                               codec='avc1')

        # End time
        end_time = time.time()

        # Calculate the duration
        total_seconds = end_time - start_time

        # Convert to minutes and seconds
        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60

        # Print the time duration
        print(f"Time duration: {minutes} minutes and {seconds:.2f} seconds")

        return annotated_images

    def create_video_from_frames(self,
                                 frames,
                                 output_video_path,
                                 fps,
                                 frame_size=None,
                                 codec='mp4v'):
        # Determine the size of the video (width, height)
        if frame_size is None:
            frame_size = (frames[0].shape[1], frames[0].shape[0])

        print('output_video_path:', output_video_path,
              'codec:', codec,
              'fps:', fps,
              'frame_size:', frame_size)

        # Initialize the VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*codec)  # Use 'mp4v' for .mp4 files
        out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

        # Write each frame to the video
        for frame in frames:
            out.write(frame)

        # Release the VideoWriter to save the video
        out.release()



straight_arrow_path = 'straight_arrow_formatted.png'
clockwise_arrow_path = 'rotated_image.png'


def test():
    # video_parser = VideoParser(straight_arrow_path,
    #                            clockwise_arrow_path,
    #                            running_mode=vision.RunningMode.IMAGE)
    video_parser = VideoParser(straight_arrow_path,
                               clockwise_arrow_path,
                               running_mode=vision.RunningMode.VIDEO)

    input_video_path = 'input_cut_rotate.mp4'
    output_video_path = 'output_cut_rotate.mp4'
    frames = video_parser.process_clockwise_video(input_video_path,
                                                  output_video_path,
                                                  is_teacher=True)

    input_video_path = 'input_cut_down.mp4'
    output_video_path = 'output_cut_down.mp4'

    frames = video_parser.process_up_down_video(input_video_path,
                                                output_video_path,
                                                is_teacher=True)

    input_video_path = 'stu_input_cut_rotate.mp4'
    output_video_path = 'stu_output_cut_rotate.mp4'
    frames = video_parser.process_clockwise_video(input_video_path,
                                                  output_video_path,
                                                  is_teacher=False)



    input_video_path = 'stu_input_cut_down.mp4'
    output_video_path = 'stu_output_cut_down.mp4'

    frames = video_parser.process_up_down_video(input_video_path,
                                                output_video_path,
                                                is_teacher=False)

    video_parser = VideoParser(straight_arrow_path,
                               clockwise_arrow_path,
                               running_mode=vision.RunningMode.IMAGE)

    frames = video_parser.process_images('teacher', 'rotate')

    frames = video_parser.process_images('stu', 'rotate')

    frames = video_parser.process_images('teacher', 'updown')

    frames = video_parser.process_images('stu', 'updown')

    concat_images_list('rotate')
    concat_images_list('updown')

    # input_video_path = '慕容横屏2.mp4'

    # seconds = [4]
    # seconds = [39, 40, 58, 59, 63, 64, 65,
    #             60 + 24, 60 + 27, 60 + 28,
    #             60 + 29, 60 + 32]

    # dest_shape = (640, 640)

    # video_parser = VideoParser(straight_arrow_path,
    #                            clockwise_arrow_path,
    #                            running_mode=vision.RunningMode.IMAGE)

    # frame_infos = video_parser.get_frames(input_video_path, seconds)

    # video_parser.save_frames(frame_infos)

    # print(len(frame_infos))

    # for image_file_name in image_file_names:
    #     video_parser.detect(image_file_name, dest_shape)

    # input_video_path = 'stu_input_cut_rotate.mp4'
    # seconds = [0]

    # video_parser = VideoParser(straight_arrow_path,
    #                            clockwise_arrow_path,
    #                            running_mode=vision.RunningMode.IMAGE)

    # frame_infos = video_parser.get_frames(input_video_path, seconds)

    # video_parser.save_frames(frame_infos)
