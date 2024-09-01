# hand_orientation_det

# Input

| 慕容横屏2.mp4 | time     |       |             |
| ------------- | -------- | ----- | ----------- |
| 劈丝          | 18-31    | 25-26 |             |
| 梳绒          | 37-45    | 39-40 |             |
| 钢丝搓紧      | 54-01:10 | 58-59 | 01:04-01:09 |

# Preprocess

## 1 Kaggle

```python
!python preprocess.py

from IPython.display import FileLink
import os

os.chdir('/kaggle/working')
json_file_path = rf'preprocess_data.json'
FileLink(str(json_file_path))

# Then download preprocess_data.json and upload it to Google Drive
```

## 2 Colab

```python
!cp /content/drive/MyDrive/data/cv/hand_orientation/*.mp4 .
!cp /content/drive/MyDrive/data/cv/hand_orientation/*.png .
!cp /content/drive/MyDrive/data/cv/hand_orientation/*.json .

!wget https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task -O gesture_recognizer.task

!wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task -O hand_landmarker.task

!pip install mediapipe

!python process_video.py
# Then download preprocess_data_001.json and video clips named like 'input_video_0001_0007.mp4'
# And put them to the public directory of frontend.
```

# Run without stu_correct

```
# Init
!mkdir -p frames
!rm frames/*


from mediapipe.tasks.python import vision


straight_arrow_path = 'straight_arrow_formatted.png'
clockwise_arrow_path = 'rotated_image.png'

video_parser = VideoParser(straight_arrow_path,
                           clockwise_arrow_path,
                           running_mode=vision.RunningMode.IMAGE)

# Teacher rotate
!rm frames/teacher_input_rotate_*.png
!rm frames/teacher_output_rotate_*.png
!ffmpeg -y -i input_video.mp4 -ss 00:01:03 -to 00:01:09 -c copy input_cut_rotate.mp4
!cp input_cut_rotate.mp4 teacher_input_rotate.mp4
!ffmpeg -y -i teacher_input_rotate.mp4 -vsync 0 frames/teacher_input_rotate_%04d.png

frames = video_parser.process_predefined_images('teacher', 'rotate')

!ffmpeg -y -framerate 29.54 -i frames/teacher_output_rotate_%04d.png -c:v libx264 -profile:v high -pix_fmt yuv420p -r 29.54 -b:v 1986k -c:a aac -b:a 128k teacher_output_rotate.mp4

# Stu rotate
!rm frames/stu_input_rotate_*.png
!rm frames/stu_output_rotate_*.png
!cp VID20240827084519.mp4 stu_input_rotate.mp4
!ffmpeg -y -i stu_input_rotate.mp4 -vsync 0 frames/stu_input_rotate_%04d.png

frames = video_parser.process_predefined_images('stu', 'rotate', False)

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

frames = video_parser.process_predefined_images('teacher', 'updown')

!ffmpeg -y -framerate 29.54 -i frames/teacher_output_updown_%04d.png -c:v libx264 -profile:v high -pix_fmt yuv420p -r 29.54 -b:v 1986k -c:a aac -b:a 128k teacher_output_updown.mp4

# Stu updown
!rm frames/stu_input_updown_*.png
!rm frames/stu_output_updown_*.png
!cp VID20240827150014.mp4 stu_input_updown.mp4
!ffmpeg -y -i stu_input_updown.mp4 -vsync 0 frames/stu_input_updown_%04d.png

frames = video_parser.process_predefined_images('stu', 'updown', False)

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

frames = video_parser.process_predefined_images('teacher', 'rotate')

!ffmpeg -y -framerate 29.54 -i frames/teacher_output_rotate_%04d.png -c:v libx264 -profile:v high -pix_fmt yuv420p -r 29.54 -b:v 1986k -c:a aac -b:a 128k teacher_output_rotate.mp4

# Stu rotate stu_correct
!rm frames/stu_input_rotate_stu_correct_*.png
!rm frames/stu_output_rotate_stu_correct_*.png
!cp VID20240828190345.mp4 stu_input_rotate_stu_correct.mp4
!ffmpeg -y -i stu_input_rotate_stu_correct.mp4 -vsync 0 frames/stu_input_rotate_stu_correct_%04d.png

frames = video_parser.process_predefined_images('stu', 'rotate')

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

frames = video_parser.process_predefined_images('teacher', 'updown')

!ffmpeg -y -framerate 29.54 -i frames/teacher_output_updown_%04d.png -c:v libx264 -profile:v high -pix_fmt yuv420p -r 29.54 -b:v 1986k -c:a aac -b:a 128k teacher_output_updown.mp4

# Stu updown stu_correct
!rm frames/stu_input_updown_stu_correct_*.png
!rm frames/stu_output_updown_stu_correct_*.png
!cp VID20240828190100.mp4 stu_input_updown_stu_correct.mp4
!ffmpeg -y -i stu_input_updown_stu_correct.mp4 -vsync 0 frames/stu_input_updown_stu_correct_%04d.png

frames = video_parser.process_predefined_images('stu', 'updown')

!ffmpeg -y -framerate 29.54 -i frames/stu_output_updown_stu_correct_%04d.png -c:v libx264 -profile:v high -pix_fmt yuv420p -r 29.54 -b:v 1986k -c:a aac -b:a 128k stu_output_updown_stu_correct.mp4

# Merge updown
!rm frames/output_updown_stu_correct_*.png

concat_images_list('updown')

!ffmpeg -y -framerate 29.54 -i frames/output_updown_stu_correct_%04d.png -c:v libx264 -profile:v high -pix_fmt yuv420p -r 29.54 -b:v 1986k -c:a aac -b:a 128k output_updown_stu_correct.mp4

echo "Completed!"
```



# 对比步骤

1. 对老师和学生视频分别进行手势检测和方向检测：

    1.1 读取视频，提取每一帧图像。

    1.2 手势检测模型对每一帧图像检测是左手还是右手，和手部的21个关键点坐标。（使用了模型）

    1.3 手势方向检测模型读取连续帧，识别哪些帧有以下四个手势之一：上，下，顺时针，逆时针。（目前人工操作，没有用模型，因为没有现成的准确率高的模型）

2. 判断老师和学生的手势判断是否一致（如上下，或者顺时针逆时针），判断学生操作是否正确。

3. 给老师和学生的手部的顶端画上手势。并使用颜色表明手势是否正确：橙色表示正确，红色表示错误。

4. 合并每一帧图像，老师的在左边显示，学生的在右边显示。
5. 从图像生成对比结果视频。



# Misc

```
# !ffmpeg -i input_cut_down.mp4 -q:a 0 -map a input_cut_down_audio.mp3
# !ffmpeg -i output_cut_down.mp4 -i input_cut_down_audio.mp3 -c:v copy -c:a aac -strict experimental output_cut_down_with_audio.mp4

# !ffmpeg -y -framerate 29.54 -i frame_%04d.png -c:v libx264 -profile:v high -pix_fmt yuv420p -r 29.54 -b:v 1986k -c:a aac -b:a 128k output_cut_rotate.mp4

# !ffmpeg -y -framerate 29.54 -i frame_%04d.png -c:v libx264 -profile:v high -pix_fmt yuv420p -r 29.54 -b:v 1986k -c:a aac -b:a 128k output_cut_down.mp4
```



```
url: http://10.68.84.28:21474/v1/chat/completions
data: 
{
    "model": "Qwen1.5-32B-Chat",
    "messages": [
        {
            "role": "system",
            "content": ""
        },
        {
            "role": "user",
            "content": "\n请从以下文本中推测制作的对象，答案只需要包含一个词，不需要解释:\n<文本>：Hello,大家好,今天小悠为大家带来的是一个新手入门的绒花教程。 首先我们选用的是苏绣的绣线,五根一组对折固定。 我们今天做的是一个扁平的款式,线可以稍微排细数一些。 接下来我们开始给这颗绣线进行披丝。 我们把每根线在根部搓粘,抽出其中一股。 这样一整根丝线就披好了。 我们把披好丝的线进行整理 接下来我们就可以用棕毛书开始进行梳容 这一步为了防止打卷 我们可以喷少量的水 或者是用湿毛巾进行擦拭 我们梳到这种没有线痕的程度就可以了 注意不要梳的太固 接下来我们用绒栓把它固定起来 就可以上童丝了 我们取一节退好火的童丝 对着之后把一端拧起来 接下来我们把铜丝上到丝线上 把另外一端也拧起来 这里注意您的方向是 右边顺时针向外 左边顺时针向里 按照这个方向两手发力 搓紧铜丝就可以了 接下来我们开始剪绒条 首先我们把绒牌堆积在手上 接下来按照视频所示的方法剪下来 剪好的绒条我们把它在桌子上堆起之后 按照之前的方向搓拧起来 接下来我们用电动橡皮给它滚融 这里按两到三下即可 这一步的工具用搓丝板或者是电钻都是可以替代的 接下来我们开始进行打尖 因为我们要做的是圆形和尖角形两种花瓣 像这种圆形的花瓣只需要给两端进行打尖就可以了 打好尖之后我们用镊子进行整理 然后弯出一个花瓣的形状 做这种尖角形的花瓣则需要在终点处再次进行打尖 这样打出两个锥形 对着之后尖角形的花瓣就做好了 接下来我们用夹板把花瓣加平 首先我们轻轻过一下花瓣 接下来再发力把它压平 为了防止它松散 我们可以使用少量的定型喷雾 接下来我们把花蕊斜切 用披好丝的绣线 把花瓣按顺序一次绑起来 再用镊子整理花瓣的形状 另外一朵花的做法也是一样的 接下来我们用珍珠做一个点缀 最后我们把所有的配件组合起来 用白胶点在收尾处固定 最后我们这朵花就做好了 今天的教程 就这么结束了 \n<制作对象>：\n"
        }
    ],
    "temperature": 0.5
}
```



```
{
    "model": "Qwen1.5-32B-Chat",
    "messages": [
        {
            "role": "system",
            "content": ""
        },
        {
            "role": "user",
            "content": "\n1: Hello,大家好,今天小悠为大家带来的是一个新手入门的绒花教程。\n2: 首先我们选用的是苏绣的绣线,五根一组对折固定。\n3: 我们今天做的是一个扁平的款式,线可以稍微排细数一些。\n4: 接下来我们开始给这颗绣线进行披丝。\n5: 我们把每根线在根部搓粘,抽出其中一股。\n6: 这样一整根丝线就披好了。\n7: 我们把披好丝的线进行整理\n8: 接下来我们就可以用棕毛书开始进行梳容\n9: 这一步为了防止打卷\n10: 我们可以喷少量的水\n11: 或者是用湿毛巾进行擦拭\n12: 我们梳到这种没有线痕的程度就可以了\n13: 注意不要梳的太固\n14: 接下来我们用绒栓把它固定起来\n15: 就可以上童丝了\n16: 我们取一节退好火的童丝\n17: 对着之后把一端拧起来\n18: 接下来我们把铜丝上到丝线上\n19: 把另外一端也拧起来\n20: 这里注意您的方向是\n21: 右边顺时针向外\n22: 左边顺时针向里\n23: 按照这个方向两手发力\n24: 搓紧铜丝就可以了\n25: 接下来我们开始剪绒条\n26: 首先我们把绒牌堆积在手上\n27: 接下来按照视频所示的方法剪下来\n28: 剪好的绒条我们把它在桌子上堆起之后\n29: 按照之前的方向搓拧起来\n30: 接下来我们用电动橡皮给它滚融\n31: 这里按两到三下即可\n32: 这一步的工具用搓丝板或者是电钻都是可以替代的\n33: 接下来我们开始进行打尖\n34: 因为我们要做的是圆形和尖角形两种花瓣\n35: 像这种圆形的花瓣只需要给两端进行打尖就可以了\n36: 打好尖之后我们用镊子进行整理\n37: 然后弯出一个花瓣的形状\n38: 做这种尖角形的花瓣则需要在终点处再次进行打尖\n39: 这样打出两个锥形\n40: 对着之后尖角形的花瓣就做好了\n41: 接下来我们用夹板把花瓣加平\n42: 首先我们轻轻过一下花瓣\n43: 接下来再发力把它压平\n44: 为了防止它松散\n45: 我们可以使用少量的定型喷雾\n46: 接下来我们把花蕊斜切\n47: 用披好丝的绣线\n48: 把花瓣按顺序一次绑起来\n49: 再用镊子整理花瓣的形状\n50: 另外一朵花的做法也是一样的\n51: 接下来我们用珍珠做一个点缀\n52: 最后我们把所有的配件组合起来\n53: 用白胶点在收尾处固定\n54: 最后我们这朵花就做好了\n55: 今天的教程\n56: 就这么结束了\n你是一位字幕校对专家，请仔细根据下面的关键词和手工常识，对上面语音转文本的字幕进行错误纠正，输出直接在原位替换错别字或者同音字即可。\n关键词：['绒花', '劈丝', '梳绒', '上铜丝', '搓紧铜丝', '剪绒条', '滚绒', '打尖', '定型']\n纠正结果：\n"
        }
    ],
    "temperature": 0.5
}
```





```
{
    "model": "Qwen1.5-32B-Chat",
    "messages": [
        {
            "role": "system",
            "content": ""
        },
        {
            "role": "user",
            "content": "\n1: Hello,大家好,今天小悠为大家带来的是一个新手入门的绒花教程。\n2: 首先我们选用的是苏绣的绣线,五根一组对折固定。\n3: 我们今天做的是一个扁平的款式,线可以稍微排细数一些。\n4: 接下来我们开始给这颗绣线进行劈丝。\n5: 我们把每根线在根部搓粘,抽出其中一股。\n6: 这样一整根丝线就劈好了。\n7: 我们把劈好丝的线进行整理\n8: 接下来我们就可以用棕毛刷开始进行梳绒\n9: 这一步为了防止打卷\n10: 我们可以喷少量的水\n11: 或者是用湿毛巾进行擦拭\n12: 我们梳到这种没有线痕的程度就可以了\n13: 注意不要梳得太紧\n14: 接下来我们用绒栓把它固定起来\n15: 就可以上铜丝了\n16: 我们取一节退好火的铜丝\n17: 对着之后把一端拧起来\n18: 接下来我们把铜丝上到丝线上\n11: 把另一端也拧起来\n19: 这里注意您的方向是\n20: 右边顺时针向外\n21: 左边顺时针向内\n22: 按照这个方向两手发力\n23: 搓紧铜丝就可以了\n24: 接下来我们开始剪绒条\n25: 首先我们把绒牌堆积在手上\n26: 接下来按照视频所示的方法剪下来\n27: 剪好的绒条我们把它在桌子上堆起之后\n28: 按照之前的方向搓拧起来\n29: 接下来我们用电动滚筒给它滚绒\n30: 这里按两到三次即可\n31: 这一步的工具可以用搓丝板或者是电钻来替代\n32: 接下来我们开始进行打尖\n33: 因为我们要做的是圆形和尖角形两种花瓣\n34: 像这种圆形的花瓣只需要给两端进行打尖就可以了\n35: 打好尖之后我们用镊子进行整理\n36: 然后弯出一个花瓣的形状\n37: 做这种尖角形的花瓣则需要在终点处再次进行打尖\n38: 这样打出两个锥形\n39: 对着之后尖角形的花瓣就做好了\n40: 接下来我们用夹板把花瓣加平\n41: 首先我们轻轻过一下花瓣\n42: 接下来再发力把它压平\n43: 为了防止它松散\n44: 我们可以使用少量的定型喷雾\n45: 接下来我们把花蕊斜切\n46: 用劈好丝的绣线\n47: 把花瓣按顺序一次绑起来\n48: 再用镊子整理花瓣的形状\n49: 另外一朵花的做法也是一样的\n50: 接下来我们用珍珠做一个点缀\n51: 最后我们把所有的配件组合起来\n52: 用白胶点在收尾处固定\n53: 最后我们这朵花就做好了\n54: 今天的教程\n55: 就这么结束了\n请从上面的文本中找到 步骤 开始和结束对应的 行 ，务必完整，输出只有两个行数数字，用英文逗号隔开，不需要解释\n步骤：劈丝\n行：\n"
        }
    ],
    "temperature": 0.5
}
```



```
[{'step': '劈丝',
  'start_line': 4,
  'start_time': '00:17',
  'start_time_str': '00:00:17,030',
  'end_time_str': '00:00:33,650',
  'end_line': 7,
  'text': '接下来我们开始给这颗绣线进行劈丝。 我们把每根线在根部搓粘,抽出其中一股。 这样一整根丝线就劈好了。 我们把劈好丝的线进行整理'},
 {'step': '梳绒',
  'start_line': 8,
  'start_time': '00:33',
  'start_time_str': '00:00:33,650',
  'end_time_str': '00:00:52,950',
  'end_line': 14,
  'text': '接下来我们就可以用棕毛刷开始进行梳绒 这一步为了防止打卷 我们可以喷少量的水 或者是用湿毛巾进行擦拭 我们梳到这种没有线痕的程度就可以了 注意不要梳的太死 接下来我们用绒栓把它固定起来'},
 {'step': '上铜丝',
  'start_line': 15,
  'start_time': '00:52',
  'start_time_str': '00:00:52,950',
  'end_time_str': '00:01:13,910',
  'end_line': 23,
  'text': '就可以上铜丝了 我们取一节退好火的铜丝 对着之后把一端拧起来 接下来我们把铜丝上到丝线上 把另外一端也拧起来 这里注意您的方向是 右边顺时针向外 左边顺时针向里 按照这个方向两手发力'},
 {'step': '搓紧铜丝',
  'start_line': 24,
  'start_time': '01:13',
  'start_time_str': '00:01:13,910',
  'end_time_str': '00:01:15,730',
  'end_line': 24,
  'text': '搓紧铜丝就可以了'},
 {'step': '剪绒条',
  'start_line': 25,
  'start_time': '01:15',
  'start_time_str': '00:01:15,730',
  'end_time_str': '00:01:36,120',
  'end_line': 30,
  'text': '接下来我们开始剪绒条 首先我们把绒牌堆积在手上 接下来按照视频所示的方法剪下来 剪好的绒条我们把它在桌子上堆起之后 按照之前的方向搓拧起来 接下来我们用电动滚筒给它滚绒'},
 {'step': '滚绒',
  'start_line': 31,
  'start_time': '01:36',
  'start_time_str': '00:01:36,120',
  'end_time_str': '00:01:52,280',
  'end_line': 34,
  'text': '这里按两到三次即可 这一步的工具可以用搓丝板或者是电钻替代 接下来我们开始进行打尖 因为我们要做的是圆形和尖角形两种花瓣'},
 {'step': '打尖',
  'start_line': 35,
  'start_time': '01:52',
  'start_time_str': '00:01:52,280',
  'end_time_str': '00:02:26,090',
  'end_line': 45,
  'text': '像这种圆形的花瓣只需要给两端进行打尖就可以了 打好尖之后我们用镊子进行整理 然后弯出一个花瓣的形状 做这种尖角形的花瓣则需要在终点处再次进行打尖 这样打出两个锥形 对着之后尖角形的花瓣就做好了 接下来我们用夹板把花瓣加平 首先我们轻轻过一下花瓣 接下来再发力把它压平 为了防止它松散 我们可以使用少量的定型喷雾'},
 {'step': '定型',
  'start_line': 46,
  'start_time': '02:26',
  'start_time_str': '00:02:26,090',
  'end_time_str': None,
  'end_line': '-1',
  'text': '接下来我们把花蕊斜切 用劈好丝的绣线 把花瓣按顺序一次绑起来 再用镊子整理花瓣的形状 另外一朵花的做法也是一样的 接下来我们用珍珠做一个点缀 最后我们把所有的配件组合起来 用白胶点在收尾处固定 最后我们这朵花就做好了 今天的教程 就这么结束了'}]
```





