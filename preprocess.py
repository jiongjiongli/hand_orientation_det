from IPython.display import FileLink
import os
from time import time
from pathlib import Path
from tqdm import tqdm
import textwrap
import json

from transformers import AutoModelForCausalLM, AutoTokenizer


def from_video_file(file_path):
    cache_dir = '/kaggle/input/mp4-text'
    cache_dir = Path(cache_dir)
    file_path = Path(file_path)
    cache_file_name = rf'{file_path.stem[-1]}{file_path.suffix}.json'
    cache_file_path = cache_dir / cache_file_name

    print('Reading:', cache_file_path)

    with open(str(cache_file_path), 'r', encoding='utf-8') as f:
        j = json.load(f)
        return j['data']


def list_to_fulltext(j):
    retstr = "".join([x['text'] + ' ' for x in j])
    return retstr


def list_to_lines(j):
    retstr = "\n".join([f'{ind+1}: ' + str(x['text']).replace('\n', ' ') for ind, x in enumerate(j)])
    return retstr


def get_crafting_object(text):
    return textwrap.dedent(f"""
请从以下文本中推测制作的对象，答案只需要包含一个词，不需要解释:
<文本>：{text}
<制作对象>：
    """)


def modify_srt(linetext, keywords):
    return textwrap.dedent(f"""
{linetext}
你是一位字幕校对专家，请仔细根据下面的关键词和手工常识，对上面语音转文本的字幕进行错误纠正，输出直接在原位替换错别字或者同音字即可。
关键词：{keywords}
纠正结果：
    """)


def get_step_lines(linetext, step):
    return textwrap.dedent(f"""
{linetext}
请从上面的文本中找到 步骤 开始和结束对应的 行 ，务必完整，输出只有两个行数数字，用英文逗号隔开，不需要解释
步骤：{step}
行：
    """)


class Model:
    def __init__(self):
        model_name = "Qwen/Qwen2-7B-Instruct"
        self.device = "cuda" # the device to load the model onto
        self.max_new_tokens=2048

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def get_response(self, message):
        start_time = time()

        data = self.create_data(message)
        messages = data['messages']

        print('=' * 40)
        print('request:')
        print(messages)
        print('-' * 40)

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        end_time = time()
        duration = int(end_time - start_time)
        minutes = duration // 60
        seconds = duration % 60
        print(rf"Duration: {minutes:02}:{seconds:02}")
        print('-' * 40)
        print('response:')
        print(response)
        print('=' * 40)

        return response

    def create_data(self, message):
        model_name = "Qwen1.5-32B-Chat"
        # system = ""
        system = "You are a helpful assistant who reads and responds in simple Chinese."
        temperature = 0.5
        data = {
            "model": model_name,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": message}],
            "temperature": temperature
        }

        return data

    def preprocess_video(self, file_path, preset):
        print('file_path:', file_path)
        file_path = Path(file_path)

        j = from_video_file(file_path)

        text_in_line = [ t['text'] for t in j ]

        preset_step = []
        vid_name = file_path.stem

        for s in preset:
            if vid_name == s['file']:
                preset_step = s['steps']

        print('preset_step:', preset_step)

        fulltext = list_to_fulltext(j)
        p = get_crafting_object(fulltext)

        obj = self.get_response(p)
        print("完成制作物品获取")

        linetext = list_to_lines(j)
        extended = [obj] + preset_step
        p = modify_srt(linetext, extended)
        linetext = self.get_response(p)
        print("完成字幕校对")

        results = []

        for s in tqdm(preset_step):
            p = get_step_lines(linetext, s)
            res = self.get_response(p)
            results.append(res)

        steps = []

        assert len(preset_step) == len(results), (len(preset_step), len(results))

        for s, res in zip(preset_step, results):
            start_line = int(res.split(',')[0])
            start_time = str(j[start_line - 1]['start_time'])
            start_time_str = start_time

            try:
                steps.append({
                    "step": s,
                    "start_line": start_line,
                    "start_time_str": start_time_str,
                })
            except Exception as ex:
                print('Exeption:', ex)
                continue
        print("完成步骤提取")


        linetexts = linetext.split('\n')

        assert len(linetexts) == len(text_in_line), (len(linetexts), len(text_in_line))

        extracted_texts = []

        for text in linetexts:
            text_parts = text.split(': ', maxsplit=1)
            extracted_text = text_parts[1]
            if  ' ' in extracted_text:
                print('special text:', extracted_text)

            extracted_text = extracted_text.replace(' ', '\n')
            extracted_texts.append(extracted_text)

        for i in range(len(steps)):
            if i < len(steps) - 1:
                steps[i]['end_time_str'] = steps[i + 1]['start_time_str']
                steps[i]['end_line'] = steps[i + 1]['start_line'] - 1
                steps[i]['text'] = ' '.join(extracted_texts[steps[i]['start_line'] - 1:steps[i]['end_line']])
            else:
                steps[i]['end_time_str'] = None
                steps[i]['end_line'] = '-1'
                steps[i]['text'] = ' '.join(extracted_texts[steps[i]['start_line'] - 1:])

        print("完成步骤文本提取")
        return steps


def main():
    file_paths = ['慕容横屏1.mp4', '慕容横屏2.mp4']

    preset = [
      {
        "file": "慕容横屏1",
        "steps": ["劈丝", "刷绒", "铜丝搓紧", "剪绒条", "滚绒", "打尖", "定型"]
      },
      {
        "file": "慕容横屏2",
        "steps": ["劈丝", "梳绒", "上铜丝", "搓紧铜丝", "剪绒条", "滚绒", "打尖", "定型"]
      },
    ]

    temp_dir = '/kaggle/working/hf_cache'
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    os.environ['HF_HOME'] = str(temp_dir)
    print('HF_HOME:', os.environ['HF_HOME'])

    model = Model()
    results = []

    for file_path in file_paths:
        steps = model.preprocess_video(file_path, preset)
        result = {
            'file_path': str(file_path),
            'steps': steps,
        }

        results.append(result)

    os.chdir('/kaggle/working')
    json_file_path = rf'preprocess_data.json'

    with open(str(json_file_path), 'w') as file_stream:
        json.dump(results, file_stream, ensure_ascii=False, indent=4)

    print('Wrote to json_file_path:', json_file_path)


if __name__ == '__main__':
    main()
