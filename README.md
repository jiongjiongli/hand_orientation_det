# hand_orientation_det





| 慕容横屏2.mp4 |          |       |             |
| ------------- | -------- | ----- | ----------- |
| 劈丝          | 18-31    | 25-26 |             |
| 梳绒          | 37-45    | 39-40 |             |
| 钢丝搓紧      | 54-01:10 | 58-59 | 01:04-01:09 |



```bash
!ffmpeg -i input_video.mp4 -ss 00:01:03 -to 00:01:09 -c copy input_cut_rotate.mp4

!ffmpeg -y -i input_video.mp4 -ss 00:00:35 -to 00:00:43 -c copy ./input_cut_down.mp4

# !ffmpeg -i input_cut_down.mp4 -q:a 0 -map a input_cut_down_audio.mp3
# !ffmpeg -i output_cut_down.mp4 -i input_cut_down_audio.mp3 -c:v copy -c:a aac -strict experimental output_cut_down_with_audio.mp4

!pip install mediapipe
!wget https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task

!wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task


!ffmpeg -y -framerate 29.54 -i frame_%04d.png -c:v libx264 -profile:v high -pix_fmt yuv420p -r 29.54 -b:v 1986k -c:a aac -b:a 128k output_cut_rotate.mp4

!ffmpeg -y -framerate 29.54 -i frame_%04d.png -c:v libx264 -profile:v high -pix_fmt yuv420p -r 29.54 -b:v 1986k -c:a aac -b:a 128k output_cut_down.mp4

```

