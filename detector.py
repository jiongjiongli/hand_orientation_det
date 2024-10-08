import itertools
import cv2
import math
from pathlib import Path
import time
from PIL import Image, ImageDraw
import subprocess
import numpy as np

from google.colab.patches import cv2_imshow
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


straight_arrow_path = 'straight_arrow_formatted.png'
clockwise_arrow_path = 'rotated_image.png'
frame_dir_path = Path('./frames')


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
            # print(f"Removed: {file_path}")
        except Exception as e:
            print(f"Error removing {file_path}: {e}")


def get_file_name(teacher_or_stu,
                  input_or_output,
                  rotate_or_updown,
                  stu_correct,
                  suffix,
                  index_format=None,
                  is_print=False):
    assert teacher_or_stu in ['teacher', 'stu'], teacher_or_stu
    assert input_or_output in ['input', 'output'], input_or_output
    assert rotate_or_updown in ['rotate', 'updown'], rotate_or_updown
    assert stu_correct in [True, False], stu_correct
    assert suffix in ['.mp4', '.png'], suffix

    if suffix in ['.png']:
        assert index_format in ['%04d', '*'], index_format

    file_name_parts = [teacher_or_stu,
                       input_or_output,
                       rotate_or_updown]

    if teacher_or_stu in ['stu'] and stu_correct:
        file_name_parts.append('stu_correct')

    if suffix in ['.png']:
        file_name_parts.append(index_format)

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


def concat_images_list(rotate_or_updown, stu_correct=True):
    image_paths_list = []

    for teacher_or_stu in ['teacher', 'stu']:
        file_name_pattern = get_file_name(teacher_or_stu,
                                                 'output',
                                                 rotate_or_updown,
                                                 stu_correct,
                                                 '.png',
                                                 index_format='*')
        image_paths = list(frame_dir_path.glob(file_name_pattern))

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


def run_command(command):
    # # Define the ffmpeg command as a string
    # command = "ffmpeg -y -framerate 29.54 -i frames/output_updown_stu_correct_%04d.png -c:v libx264 -profile:v high -pix_fmt yuv420p -r 29.54 -b:v 1986k -c:a aac -b:a 128k output_updown_stu_correct.mp4"
    # command = "ffmpeg -y -i teacher_input_rotate.mp4 -vsync 0 frames/teacher_input_rotate_%04d.png"

    # Run the command and capture output
    try:
        print("Running command:")
        print(command)
        result = subprocess.run(command, shell=True, check=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        # Print the captured output
        print("Output:")
        print(result.stdout)
        print(result.stderr)
        print("Video created successfully!")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while creating the video: {e}")
        print("Output:\n", e.stdout)
        print("Error:\n", e.stderr)


class VideoParser:
    def __init__(self,
                 straight_arrow_path,
                 clockwise_arrow_path,
                 running_mode):
        self.clockwise_arrow_path = clockwise_arrow_path
        self.straight_arrow_path = straight_arrow_path
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

    def detect(self, image_path, teacher_or_stu, rotate_or_updown, stu_correct):
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

            if teacher_or_stu in ['teacher'] or (teacher_or_stu in ['stu'] and stu_correct):
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

            background_image = annotated_image
            min_x = np.min([landmark.x for landmark in hand_landmarks]) * background_image.width
            max_x = np.max([landmark.x for landmark in hand_landmarks]) * background_image.width

            max_dest_width = min(int(max_x + 1 - min_x), background_image.width // 5)
            max_dest_height = background_image.height // 5
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
            background_image.paste(resized_foreground_image, bbox, mask=mask)

        return annotated_image

    def process_images(self,
                       input_file_path_pattern,
                       output_dir,
                       teacher_or_stu,
                       rotate_or_updown,
                       stu_correct=True):
        # Start recording time
        start_time = time.time()

        annotated_images = []

        image_paths = list(input_file_path_pattern.parent.glob(input_file_path_pattern.name))
        image_paths.sort()

        for image_path in image_paths:
            annotated_image = self.detect(image_path,
                                          teacher_or_stu,
                                          rotate_or_updown,
                                          stu_correct)
            frame_file_path = output_dir / image_path.name.replace('input', 'output')
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

    def process_predefined_images(self, teacher_or_stu, rotate_or_updown, stu_correct=True):
        input_file_name_pattern = get_file_name(teacher_or_stu,
                                                'input',
                                                rotate_or_updown,
                                                stu_correct,
                                                '.png',
                                                index_format='*')
        input_file_path_pattern = frame_dir_path / input_file_name_pattern
        output_dir = frame_dir_path
        self.process_images(input_file_path_pattern,
                            output_dir,
                            teacher_or_stu,
                            rotate_or_updown,
                            stu_correct=stu_correct)
        return annotated_images


def get_input_images(teacher_or_stu, input_video_path):
    assert teacher_or_stu in ['teacher', 'stu'], teacher_or_stu

    # input_video_path = 'teacher_input_video.mp4'
    input_image_name_pattern = rf'{teacher_or_stu}_input_image_%04d.png'
    input_image_search_pattern = rf'{teacher_or_stu}_input_image_*.png'
    remove_files_with_pattern(frame_dir_path, input_image_search_pattern)

    input_video_path = Path(input_video_path)
    assert input_video_path.exists(), input_video_path

    input_image_path_pattern = frame_dir_path / input_image_name_pattern
    command = rf"ffmpeg -y -i {input_video_path} -vsync 0 {input_image_path_pattern}"
    run_command(command)

    print('Completed!')
    return input_image_name_pattern


def get_output_images(input_video_path, input_image_name_pattern, parse_results, teacher_or_stu, action, same_action):
    assert action in ['up', 'down', 'clockwise', 'counter_clockwise'], action
    output_image_search_pattern = rf'{teacher_or_stu}_output_image_*.png'
    remove_files_with_pattern(frame_dir_path, output_image_search_pattern)

    video_parser.get_output_images(parse_results, teacher_or_stu, action, same_action)
    print('Completed!')

def merge_images(teacher_input_video_path, stu_input_video_path, teacher_action, stu_action):
    output_image_name_pattern = rf'output_image_%04d.png'
    output_image_search_pattern = rf'output_image_*.png'
    output_video_path = 'output_video.mp4'
    remove_files_with_pattern(frame_dir_path, output_image_search_pattern)
    concat_images_list()


def test():
    frame_dir_path.mkdir(parents=True, exist_ok=True)
    # remove_files_with_pattern(frame_dir_path, '*.*')
    input_video_dict = {
        'teacher': 'teacher_input_video.mp4',
        'stu': 'stu_input_video.mp4'
    }

    info_dict = {}

    for teacher_or_stu, input_video_path in input_video_dict.items():
        input_image_name_pattern = get_input_images(teacher_or_stu, input_video_path)
        det_results = video_parser.predict_landmarks(frame_dir_path, input_image_name_pattern)
        action = video_parser.predict_action(det_results)

        info_dict[teacher_or_stu] = {
            'input_video_path': input_video_path,
            'input_image_name_pattern': input_image_name_pattern,
            'parse_results': parse_results,
            'action': action
        }

    same_action = info_dict['teacher']['action'] == info_dict['stu']['action']

    for teacher_or_stu, info in info_dict.items():
        get_output_images(info['input_video_path'],
                          info['input_image_name_pattern'],
                          info['parse_results'],
                          teacher_or_stu,
                          info['action'],
                          same_action)

        # Generate video
        output_video_path = info['input_video_path'].parent / info['input_video_path'].name.replace('input', 'output')

        output_image_path_pattern = frame_dir_path / info['input_image_name_pattern'].replace('input', 'output')
        command = rf"ffmpeg -y -framerate 29.54 -i {output_image_path_pattern} -c:v libx264 -profile:v high -pix_fmt yuv420p -r 29.54 -b:v 1986k -c:a aac -b:a 128k {output_video_path}"
        run_command(command)

    merge_images(input_video_dict['teacher'], input_video_dict['stu'], info_dict['teacher']['action'], info_dict['stu']['action'])

    # Generate video
    output_image_path_pattern = frame_dir_path / output_image_name_pattern
    command = rf"ffmpeg -y -framerate 29.54 -i {output_image_path_pattern} -c:v libx264 -profile:v high -pix_fmt yuv420p -r 29.54 -b:v 1986k -c:a aac -b:a 128k {output_video_path}"
    run_command(command)

    print('Completed!')


    video_parser = VideoParser(straight_arrow_path,
                               clockwise_arrow_path,
                               running_mode=vision.RunningMode.IMAGE)

    video_parser.process_images('teacher', 'rotate')
    video_parser.process_images('teacher', 'updown')
    video_parser.process_images('stu', 'rotate', False)
    video_parser.process_images('stu', 'updown', False)
    video_parser.process_images('stu', 'rotate')
    video_parser.process_images('stu', 'updown')

    concat_images_list('rotate')
    concat_images_list('updown')
    concat_images_list('rotate', False)
    concat_images_list('updown', False)


def main():
    file_name_parts = [['teacher', 'stu'],
    ['input', 'output'],
    ['rotate', 'updown'],
    [True, False], # stu_correct
    ['.mp4', '.png'],
    ['%04d', '*']]

    combinations = list(itertools.product(*file_name_parts))

    for combination in combinations:
        print(combination)
        get_file_name(*combination, is_print=True)


if __name__ == '__main__':
    main()
