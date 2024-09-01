import os
import json
from time import time
from pathlib import Path
import shutil
import subprocess
from tqdm import tqdm
import math
import re


frame_dir_path = Path('./frames')
frame_dir_path.mkdir(parents=True, exist_ok=True)


def parse_timestamp(time_str):
    time_str = time_str.replace(',', '.')
    timestamp_parts = time_str.split(':')
    timestamp = 0.0

    for timestamp_part in timestamp_parts:
        timestamp = timestamp * 60 + float(timestamp_part)

    timestamp = round(timestamp, 3)

    return timestamp


def format_timestamp(timestamp):
    # Ensure the timestamp has three decimal places
    timestamp = round(timestamp, 3)
    timestamp_str = rf"{timestamp:.3f}"
    timestamp_str_parts = timestamp_str.split('.')
    assert len(timestamp_str_parts) == 2, timestamp_str_parts
    milliseconds_str_part = timestamp_str_parts[1]

    # Get the total seconds as an integer
    total_seconds = int(timestamp_str_parts[0])

    # Calculate hours, minutes, and seconds
    total_minutes = total_seconds // 60
    seconds = total_seconds % 60

    hours = total_minutes // 60
    minutes = total_minutes % 60

    # Format the timestamp as "hh:mm:ss.sss"
    formatted_timestamp = f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds_str_part}"
    return formatted_timestamp


def run_command(command, parse_log_func=None, print_log=False):
    # # Define the ffmpeg command as a string
    # command = "ffmpeg -y -framerate 29.54 -i frames/output_updown_stu_correct_%04d.png -c:v libx264 -profile:v high -pix_fmt yuv420p -r 29.54 -b:v 1986k -c:a aac -b:a 128k output_updown_stu_correct.mp4"
    # command = "ffmpeg -y -i teacher_input_rotate.mp4 -vsync 0 frames/teacher_input_rotate_%04d.png"

    # Run the command and capture output
    if print_log:
        print("Running command:")
        print(command)

    # Execute the command
    process = subprocess.Popen(command.split(),
                               stderr=subprocess.STDOUT,
                               stdout=subprocess.PIPE,
                               text=True)

    while True:
        # Read each line until EOF
        line = process.stdout.readline()
        if not line:
            break

        if print_log:
            # Print the captured output
            print(line)

        if parse_log_func is not None:
            parse_log_func(line)

    # Wait for the process to complete
    process.wait()

    # Handle errors if any
    if process.returncode != 0:
        raise Exception(f"Command failed with exit code {process.returncode}")

    else:
        if print_log:
            print("Command run successfully!")


def extract_frame_timestamp_infos(video_file_path):
    # Regular expression to match frame number and pts_time entries
    frame_timestamp_info_pattern = re.compile(r"n:\s*([0-9]+).*pts_time:\s*([0-9.]+)")

    # Real-time processing of stderr
    frame_timestamp_infos = []

    def parse_frame_timestamp(line):
        match = frame_timestamp_info_pattern.search(line)

        if match:
            n, pts_time = match.groups()
            frame_index = int(n)
            frame_timestamp = float(pts_time)
            frame_timestamp = round(frame_timestamp, 3)
            frame_timestamp_infos.append((frame_index, frame_timestamp))
            # print(f"Frame {n}: pts_time = {pts_time}")  # Print each matched frame info in real-time

    # Run the ffmpeg command with showinfo filter to capture frame information
    command = rf"ffmpeg -i {video_file_path} -vf showinfo -vsync 0 -f null -"
    run_command(command, parse_log_func=parse_frame_timestamp)
    return frame_timestamp_infos


def find_frames(frame_timestamp_infos, start_time, end_time):
    frame_indexes = []

    for frame_index, frame_timestamp in frame_timestamp_infos:
        if start_time <= frame_timestamp <= end_time:
            frame_indexes.append(frame_index)

    return frame_indexes


def verify_frame_indexes(frame_indexes):
    min_frame_index = min(frame_indexes)
    expected_frame_index = min_frame_index

    for frame_index in frame_indexes:
        assert frame_index == expected_frame_index, (frame_index, expected_frame_index, frame_indexes)
        expected_frame_index += 1


def main():
    start_time_sec = time()
    json_file_path = rf'preprocess_data.json'
    json_file_path = Path(json_file_path)

    with open(str(json_file_path), 'r') as file_stream:
        preprocessed_data = json.load(file_stream)

    for file_index, file_info in enumerate(preprocessed_data):
        file_path = file_info['file_path']
        file_path = Path(file_path)

        if not file_path.exists():
            print('Ignore non-exist file:', file_path)
            continue

        print('Processing:', file_path)
        input_video_path = rf'input_video_{file_index:04d}{file_path.suffix}'
        input_video_path = Path(input_video_path)

        shutil.copy(str(file_path), str(input_video_path))

        frame_timestamp_infos = extract_frame_timestamp_infos(input_video_path)

        image_path_pattern = frame_dir_path / rf'{input_video_path.stem}' / 'all' / rf"%04d.png"
        image_path_pattern.parent.mkdir(parents=True, exist_ok=True)

        print('Extracting frames...')
        command = rf"ffmpeg -y -i {input_video_path} -vsync 0 {image_path_pattern}"
        run_command(command)
        print('Processing steps...')

        step_infos = file_info['steps']

        for step_index, step_info in enumerate(tqdm(step_infos)):
            start_time_str = step_info.get('start_time_str')
            end_time_str = step_info.get('end_time_str')

            if start_time_str is None:
                print('Ignore step because start_time_str is None:', step_info)
                continue

            if not ((end_time_str is None) or (start_time_str < end_time_str)):
                print('Ignore step because of wrong time:', step_info)
                continue

            start_time = parse_timestamp(start_time_str)

            if end_time_str is not None:
                end_time = parse_timestamp(end_time_str)
            else:
                end_time = max(frame_timestamp for _, frame_timestamp in frame_timestamp_infos)

            frame_indexes = find_frames(frame_timestamp_infos, start_time, end_time)

            if not frame_indexes:
                print('Ignore step because of empty frame_indexes:', step_info)
                continue

            frame_indexes.sort()

            verify_frame_indexes(frame_indexes)

            min_frame_index = min(frame_indexes)
            max_frame_index = max(frame_indexes)

            step_info['min_frame_index'] = min_frame_index
            step_info['max_frame_index'] = max_frame_index
            step_info['start_time'] = format_timestamp(start_time)
            step_info['end_time'] = format_timestamp(end_time)

            step_input_dir = frame_dir_path / rf'{input_video_path.stem}' / rf'{step_index:04d}' / 'input'

            for frame_index in frame_indexes:
                frame_id = frame_index + 1
                src_image_path = image_path_pattern.parent / rf"{frame_id:04d}.png"
                dest_image_path = step_input_dir / src_image_path.name
                dest_image_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src_image_path, dest_image_path)

            step_input_video_path = rf'{input_video_path.stem}_{step_index:04d}{input_video_path.suffix}'
            step_input_video_path = Path(step_input_video_path)

            step_image_path_pattern = step_input_dir / image_path_pattern.name
            step_image_path_pattern.parent.mkdir(parents=True, exist_ok=True)
            command = rf"ffmpeg -y -framerate 29.54 -start_number {min_frame_index} -i {step_image_path_pattern} -c:v libx264 -profile:v high -pix_fmt yuv420p -r 29.54 -b:v 1986k -c:a aac -b:a 128k {step_input_video_path}"

            run_command(command)

            assert step_input_video_path.exists(), step_input_video_path
            step_info['input_video_path'] = str(step_input_video_path)

            input_file_path_pattern = step_input_dir / rf'*{image_path_pattern.suffix}'

            step_output_dir = step_input_dir.parent / 'output'
            step_output_dir.mkdir(parents=True, exist_ok=True)

            step_info['input_file_path_pattern'] = str(input_file_path_pattern)
            step_info['step_output_dir'] = str(step_output_dir)

            # if step_index == 1:
            #     frames = video_parser.process_images(input_file_path_pattern, step_output_dir, 'teacher', 'updown')
            # if step_index == 2:
            #     frames = video_parser.process_images(input_file_path_pattern, step_output_dir, 'teacher', 'rotate')

        print('=' * 40)
        print('Generated video files:')
        print('-' * 40)

        for step_info in step_infos:
            if 'input_video_path' in step_info:
                step_name = step_info['step']
                step_input_video_path = step_info['input_video_path']
                print(step_name, step_input_video_path)

        print('=' * 40)

    data_file_path = rf'preprocess_data_001.json'

    with open(str(data_file_path), 'w') as file_stream:
        json.dump(preprocessed_data, file_stream, ensure_ascii=False, indent=4)

    print('Wrote to data_file_path:', data_file_path)
    print('Completed!')
    end_time_sec = time()

    duration = int(end_time_sec - start_time_sec)
    minutes = duration // 60
    seconds = duration % 60
    print(rf"Duration: {minutes:02}:{seconds:02}")


if __name__ == '__main__':
    main()
