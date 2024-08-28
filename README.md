# hand_orientation_det



# Input

| 慕容横屏2.mp4 | time     |       |             |
| ------------- | -------- | ----- | ----------- |
| 劈丝          | 18-31    | 25-26 |             |
| 梳绒          | 37-45    | 39-40 |             |
| 钢丝搓紧      | 54-01:10 | 58-59 | 01:04-01:09 |

# Preparation

```bash
cp /content/drive/MyDrive/data/cv/hand_orientation/慕容横屏2.mp4 ./input_video.mp4

!pip install mediapipe

!wget https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task

!wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task


# !ffmpeg -i input_cut_down.mp4 -q:a 0 -map a input_cut_down_audio.mp3
# !ffmpeg -i output_cut_down.mp4 -i input_cut_down_audio.mp3 -c:v copy -c:a aac -strict experimental output_cut_down_with_audio.mp4

# !ffmpeg -y -framerate 29.54 -i frame_%04d.png -c:v libx264 -profile:v high -pix_fmt yuv420p -r 29.54 -b:v 1986k -c:a aac -b:a 128k output_cut_rotate.mp4

# !ffmpeg -y -framerate 29.54 -i frame_%04d.png -c:v libx264 -profile:v high -pix_fmt yuv420p -r 29.54 -b:v 1986k -c:a aac -b:a 128k output_cut_down.mp4

```

# Run

```
# Init
!mkdir -p frames
!rm frames/*

video_parser = VideoParser(straight_arrow_path,
                           clockwise_arrow_path,
                           running_mode=vision.RunningMode.IMAGE)

# Teacher rotate
!rm frames/teacher_input_rotate_*.png
!rm frames/teacher_output_rotate_*.png
!ffmpeg -y -i input_video.mp4 -ss 00:01:03 -to 00:01:09 -c copy input_cut_rotate.mp4
!cp input_cut_rotate.mp4 teacher_input_rotate.mp4
!ffmpeg -y -i teacher_input_rotate.mp4 -vsync 0 frames/teacher_input_rotate_%04d.png

frames = video_parser.process_images('teacher', 'rotate')

!ffmpeg -y -framerate 29.54 -i frames/teacher_output_rotate_%04d.png -c:v libx264 -profile:v high -pix_fmt yuv420p -r 29.54 -b:v 1986k -c:a aac -b:a 128k teacher_output_rotate.mp4

# Stu rotate
!rm frames/stu_input_rotate_*.png
!rm frames/stu_output_rotate_*.png
!cp VID20240827084519.mp4 stu_input_rotate.mp4
!ffmpeg -y -i stu_input_rotate.mp4 -vsync 0 frames/stu_input_rotate_%04d.png

frames = video_parser.process_images('stu', 'rotate')

!ffmpeg -y -framerate 29.54 -i frames/stu_output_rotate_%04d.png -c:v libx264 -profile:v high -pix_fmt yuv420p -r 29.54 -b:v 1986k -c:a aac -b:a 128k stu_output_rotate.mp4

# Merge rotate
!rm frames/output_rotate*

concat_images_list('rotate')

!ffmpeg -y -framerate 29.54 -i frames/output_rotate_%04d.png -c:v libx264 -profile:v high -pix_fmt yuv420p -r 29.54 -b:v 1986k -c:a aac -b:a 128k output_rotate.mp4

# Teacher updown
!rm frames/teacher_input_updown_*.png
!rm frames/teacher_output_updown_*.png
!ffmpeg -y -i input_video.mp4 -ss 00:00:35 -to 00:00:43 -c copy input_cut_down.mp4
!cp input_cut_down.mp4 teacher_input_updown.mp4
!ffmpeg -y -i teacher_input_updown.mp4 -vsync 0 frames/teacher_input_updown_%04d.png

frames = video_parser.process_images('teacher', 'updown')

!ffmpeg -y -framerate 29.54 -i frames/teacher_output_updown_%04d.png -c:v libx264 -profile:v high -pix_fmt yuv420p -r 29.54 -b:v 1986k -c:a aac -b:a 128k teacher_output_updown.mp4

# Stu updown
!rm frames/stu_input_updown_*.png
!rm frames/stu_output_updown_*.png
!cp VID20240827150014.mp4 stu_input_updown.mp4
!ffmpeg -y -i stu_input_updown.mp4 -vsync 0 frames/stu_input_updown_%04d.png

frames = video_parser.process_images('stu', 'updown')

!ffmpeg -y -framerate 29.54 -i frames/stu_output_updown_%04d.png -c:v libx264 -profile:v high -pix_fmt yuv420p -r 29.54 -b:v 1986k -c:a aac -b:a 128k stu_output_updown.mp4

# Merge updown
!rm frames/output_updown*

concat_images_list('updown')

!ffmpeg -y -framerate 29.54 -i frames/output_updown_%04d.png -c:v libx264 -profile:v high -pix_fmt yuv420p -r 29.54 -b:v 1986k -c:a aac -b:a 128k output_updown.mp4
```

