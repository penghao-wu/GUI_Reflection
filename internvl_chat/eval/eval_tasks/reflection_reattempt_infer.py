import os
import json
import torch
import random
import argparse
import itertools
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import ast
import re

from infer import Model
from utils import read_jsonl_file, read_json_file, write_jsonl_file, open_image, draw_bbox_to_image_pil, InferenceSampler

prompt_format_first_round = """<image>\nFor the operation: <ref>{}</ref>, output the bounding box of the area in the image that can complete the task."""

prompt_format_multi_round = """<image>\nYou are given a screenshot of a mobile phone screen, a question, and some incorrect answers that are already excluded for you.
You need to give a correct answer based on the screenshot. Notice that the correct answer should be different from the incorrect answers.
The question is: {question}
The incorrect answers (also annotated using red bbox in the image) are: {incorrect_answer}
The correct answer is:
"""


def pred_2_bbox(s):
    match = re.search(r'\[(\d+,\s*\d+,\s*\d+,\s*\d+)\]', s)

    if match:
        extracted_content = match.group(1)
        extracted_list = ast.literal_eval(f'[{extracted_content}]')
    floats = [int(num) for num in extracted_list]

    return floats

def pred_2_point(s):
    match = re.search(r'\[(\d+,\s*\d+,\s*\d+,\s*\d+)\]', s)

    if match:
        extracted_content = match.group(1)
        extracted_list = ast.literal_eval(f'[{extracted_content}]')
    floats = [float(num) for num in extracted_list]
    # Convert the tuples of strings into tuples of integers
    if len(floats) == 2:
        click_point = floats
    elif len(floats) == 4:
        click_point = [(floats[0]+floats[2])/2, (floats[1]+floats[3])/2]
    return click_point

def calculate_accuracy(stats_dict):
    results = {}
    for task, values in stats_dict.items():
        text_acc = sum(values["text"]) / len(values["text"]) if values["text"] else 0
        icon_acc = sum(values["icon"]) / len(values["icon"]) if values["icon"] else 0
        results[task] = [text_acc, icon_acc]
    return results

def evaluate_predictions(all_data, current_round):
    task_results = defaultdict(lambda: {"text": [], "icon": []})
    correct_pred = 0
    num_wrong_format = 0
    for item in all_data:
        task = item['task']
        data_type = item['data_type']
        gt_bbox = item['bbox_normalized']

        if current_round > 1:
            correct_flag = False
            for round_i in range(1, current_round):
                if item[f"is_correct_round{round_i}"]:
                    correct_flag = True
                    break
            if correct_flag:
                correct_pred += 1
                task_results[task][data_type].append(1)
                continue
        try:
            result = item[f'pred_response_round{current_round}']
            pred_point = pred_2_point(result)
            pred_point = [v / 1000 for v in pred_point]
            pred_bbox = pred_2_bbox(result)
            is_correct = gt_bbox[0] <= pred_point[0] <= gt_bbox[2] and gt_bbox[1] <= pred_point[1] <= gt_bbox[3]
            if is_correct:
                task_results[task][data_type].append(1)
                item[f"is_correct_round{current_round}"] = 1
            else:
                task_results[task][data_type].append(0)
                if 'negative_bboxes' not in item:
                    item['negative_bboxes'] = []
                item['negative_bboxes'].append(pred_bbox)
                item[f"is_correct_round{current_round}"] = 0
        except Exception as e:
            num_wrong_format += 1
            print('wrong format:', result)
            task_results[task][data_type].append(0)
            item[f"is_correct_round{current_round}"] = 0
    
    print("Total num:", len(all_data))
    print("Wrong format num:", num_wrong_format)
    acc_results = calculate_accuracy(task_results)
    for task, (text_acc, icon_acc) in acc_results.items():
        print(f"[{task}] Text Acc: {text_acc:.2%}, Icon Acc: {icon_acc:.2%}")

    text_acc_all = np.mean([value[0] for value in acc_results.values()])
    icon_acc_all = np.mean([value[1] for value in acc_results.values()])
    acc_all = (text_acc_all + icon_acc_all)/2
    print(f"Avg Text Acc: {text_acc_all:.2%}, Avg Icon Acc: {icon_acc_all:.2%}, Avg Total Acc: {acc_all:.2%}")
    return all_data


def parse_args():
    parser = argparse.ArgumentParser(description="InternVL-eval")
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--image_root', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--out_dir', type=str, default='./result/mistake_reattempt')
    parser.add_argument('--result_filename', type=str, default='')
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--max_num', type=int, default=12)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--inference_round', type=int, default=1)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    random.seed(args.seed)

    model_path = args.checkpoint
    image_root = args.image_root

    if args.result_filename == '':
        result_filename = model_path.rstrip(os.sep).split(os.sep)[-1] + '.jsonl'
    else:
        result_filename = args.result_filename

    result_filename = f"round{args.inference_round}_" + result_filename
    
    jsonl_data = read_jsonl_file(args.data_path)
    all_data = jsonl_data
    for i, data in enumerate(all_data):
        data['index'] = i

    os.makedirs(args.out_dir, exist_ok=True)
    result_path = os.path.join(args.out_dir, result_filename)
    print(result_path)

    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": True if args.temperature else False,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "num_beams": args.num_beams,
    }

    # Initialize distributed training
    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )
    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    dataset_size = len(all_data)
    sampler = InferenceSampler(dataset_size)

    model = Model(model_path, args.max_num, auto=False)

    outputs = []
    for idx in tqdm(sampler):
        data_item = all_data[idx]
        image = data_item['image']

        negative_bboxes= data_item.get('negative_bboxes', [])
        instruction = data_item['instruction']

        if args.inference_round == 1:
            question = prompt_format_first_round.format(instruction)
                
        else:
            question = prompt_format_multi_round.format(question=instruction, incorrect_answer=str(negative_bboxes))

        if args.inference_round > 1:
            correct_flag = False
            for round_i in range(1, args.inference_round):
                if data_item[f"is_correct_round{round_i}"]:
                    correct_flag = True
                    break
            if correct_flag:
                outputs.append(data_item)
                continue

        image = open_image(os.path.join(image_root, image))
        img_size = image.size
        gt_bbox = data_item['bbox']
        gt_bbox = [gt_bbox[0], gt_bbox[1], gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]]
        data_item['bbox_normalized'] = [gt_bbox[0] / img_size[0], gt_bbox[1] / img_size[1], gt_bbox[2] / img_size[0], gt_bbox[3] / img_size[1]]
        if args.inference_round > 1:
            image = draw_bbox_to_image_pil(image, negative_bboxes)

        response = model(question, [image], generation_config)

        pred_data = data_item
        pred_data[f'pred_response_round{args.inference_round}'] = response
        outputs.append(pred_data)

    # Gather results from all processes
    torch.distributed.barrier()
    world_size = torch.distributed.get_world_size()
    merged_outputs = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))
    

    # Combine outputs
    merged_outputs = [json.loads(o) for o in merged_outputs]
    merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

    # Save results only on the main rank
    if torch.distributed.get_rank() == 0:
        merged_outputs = evaluate_predictions(merged_outputs, args.inference_round)
        write_jsonl_file(merged_outputs, result_path)

if __name__ == '__main__':
    main()