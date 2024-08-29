# hand_orientation_det

# Input

| 慕容横屏2.mp4 | time     |       |             |
| ------------- | -------- | ----- | ----------- |
| 劈丝          | 18-31    | 25-26 |             |
| 梳绒          | 37-45    | 39-40 |             |
| 钢丝搓紧      | 54-01:10 | 58-59 | 01:04-01:09 |

# Preparation

```bash
!cp /content/drive/MyDrive/data/cv/hand_orientation/*.mp4 .
!cp /content/drive/MyDrive/data/cv/hand_orientation/*.png .
!cp 慕容横屏2.mp4 ./input_video.mp4

!pip install mediapipe

!wget https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task

!wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task


# !ffmpeg -i input_cut_down.mp4 -q:a 0 -map a input_cut_down_audio.mp3
# !ffmpeg -i output_cut_down.mp4 -i input_cut_down_audio.mp3 -c:v copy -c:a aac -strict experimental output_cut_down_with_audio.mp4

# !ffmpeg -y -framerate 29.54 -i frame_%04d.png -c:v libx264 -profile:v high -pix_fmt yuv420p -r 29.54 -b:v 1986k -c:a aac -b:a 128k output_cut_rotate.mp4

# !ffmpeg -y -framerate 29.54 -i frame_%04d.png -c:v libx264 -profile:v high -pix_fmt yuv420p -r 29.54 -b:v 1986k -c:a aac -b:a 128k output_cut_down.mp4

```

# Run without stu_correct

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

frames = video_parser.process_images('stu', 'rotate', False)

!ffmpeg -y -framerate 29.54 -i frames/stu_output_rotate_%04d.png -c:v libx264 -profile:v high -pix_fmt yuv420p -r 29.54 -b:v 1986k -c:a aac -b:a 128k stu_output_rotate.mp4

# Merge rotate
!rm frames/output_rotate_*

concat_images_list('rotate', False)

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

frames = video_parser.process_images('stu', 'updown', False)

!ffmpeg -y -framerate 29.54 -i frames/stu_output_updown_%04d.png -c:v libx264 -profile:v high -pix_fmt yuv420p -r 29.54 -b:v 1986k -c:a aac -b:a 128k stu_output_updown.mp4

# Merge updown
!rm frames/output_updown_*

concat_images_list('updown', False)

!ffmpeg -y -framerate 29.54 -i frames/output_updown_%04d.png -c:v libx264 -profile:v high -pix_fmt yuv420p -r 29.54 -b:v 1986k -c:a aac -b:a 128k output_updown.mp4

echo "Completed!"
```

# Run with stu_correct

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

# Stu rotate stu_correct
!rm frames/stu_input_rotate_stu_correct_*.png
!rm frames/stu_output_rotate_stu_correct_*.png
!cp VID20240828190345.mp4 stu_input_rotate_stu_correct.mp4
!ffmpeg -y -i stu_input_rotate_stu_correct.mp4 -vsync 0 frames/stu_input_rotate_stu_correct_%04d.png

frames = video_parser.process_images('stu', 'rotate')

!ffmpeg -y -framerate 29.54 -i frames/stu_output_rotate_stu_correct_%04d.png -c:v libx264 -profile:v high -pix_fmt yuv420p -r 29.54 -b:v 1986k -c:a aac -b:a 128k stu_output_rotate_stu_correct.mp4

# Merge rotate
!rm frames/output_rotate_stu_correct_*.png

concat_images_list('rotate')

!ffmpeg -y -framerate 29.54 -i frames/output_rotate_stu_correct_%04d.png -c:v libx264 -profile:v high -pix_fmt yuv420p -r 29.54 -b:v 1986k -c:a aac -b:a 128k output_rotate_stu_correct.mp4

# Teacher updown
!rm frames/teacher_input_updown_*.png
!rm frames/teacher_output_updown_*.png
!ffmpeg -y -i input_video.mp4 -ss 00:00:35 -to 00:00:43 -c copy input_cut_down.mp4
!cp input_cut_down.mp4 teacher_input_updown.mp4
!ffmpeg -y -i teacher_input_updown.mp4 -vsync 0 frames/teacher_input_updown_%04d.png

frames = video_parser.process_images('teacher', 'updown')

!ffmpeg -y -framerate 29.54 -i frames/teacher_output_updown_%04d.png -c:v libx264 -profile:v high -pix_fmt yuv420p -r 29.54 -b:v 1986k -c:a aac -b:a 128k teacher_output_updown.mp4

# Stu updown stu_correct
!rm frames/stu_input_updown_stu_correct_*.png
!rm frames/stu_output_updown_stu_correct_*.png
!cp VID20240828190100.mp4 stu_input_updown_stu_correct.mp4
!ffmpeg -y -i stu_input_updown_stu_correct.mp4 -vsync 0 frames/stu_input_updown_stu_correct_%04d.png

frames = video_parser.process_images('stu', 'updown')

!ffmpeg -y -framerate 29.54 -i frames/stu_output_updown_stu_correct_%04d.png -c:v libx264 -profile:v high -pix_fmt yuv420p -r 29.54 -b:v 1986k -c:a aac -b:a 128k stu_output_updown_stu_correct.mp4

# Merge updown
!rm frames/output_updown_stu_correct_*.png

concat_images_list('updown')

!ffmpeg -y -framerate 29.54 -i frames/output_updown_stu_correct_%04d.png -c:v libx264 -profile:v high -pix_fmt yuv420p -r 29.54 -b:v 1986k -c:a aac -b:a 128k output_updown_stu_correct.mp4

echo "Completed!"
```



# 对比步骤

1. 对老师和学生视频分别进行手势检测：

    1.1 读取视频，提取每一帧图像。

    1.2 手势检测模型对每一帧图像检测是左手还是右手，和手部的21个关键点坐标。（使用了模型）

    1.3 手势方向检测模型读取连续帧，识别哪些帧有以下四个手势之一：上，下，顺时针，逆时针。（目前人工操作，没有用模型，因为没有现成的准确率高的模型）

2. 判断老师和学生的手势判断是否一致（如上下，或者顺时针逆时针），判断学生操作是否正确。

3. 给手部的顶端画上手势。并使用颜色表明手势是否正确：橙色表示正确，红色表示错误。

4. 合并所有的对比的帧，生成对比结果视频。
