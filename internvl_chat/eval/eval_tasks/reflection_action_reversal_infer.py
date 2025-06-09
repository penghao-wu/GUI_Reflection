import os
import json
import torch
import random
import argparse
import itertools
from tqdm import tqdm
import numpy as np

from eval.infer import Model
from eval.utils import read_jsonl_file, write_jsonl_file, open_image, InferenceSampler, apply_click_to_image_pil


prompt_format = """<image>\n<image>\nYou are an expert in evaluating the behavior of GUI agents that interact with Android phone interfaces. Your task is to assist in training intelligent agents by identifying the correct **revert operation** to undo a previously executed incorrect operation.

You will be given:

- A **current action** that was performed
- A pair of screenshots:  
  - **Screen A**: the UI before the action  
  - **Screen B**: the UI after the action  
- A set of **six revert operation action choices (A–F)**  
- The agent must choose **the one correct revert operation** that best reverts Screen B back to Screen A.

---

## Valid Action Space:
- **Open app[app]**: Open the specified app.  
- **Click**: Tap on a specific UI element.  
- **Long Press**: Long press on a specific UI element.  
- **Scroll**: Perform a scroll gesture on the screen.  
- **Type[text]**: Input the specified text into a text field.  
- **Press Home**: Return to the home screen.  
- **Press Back**: Return to the previous screen.  
- **Press Enter**: Confirm input using the enter/return key.

---

## Evaluation Criteria:
- The **correct revert operation** must be the most effective and reasonable way to return the UI from **Screen B** to **Screen A**, based on the change caused by the current action.
- Only **one option** is correct. The remaining five should be plausible but incorrect.
- Evaluate the options based on:
  - Whether the action targets the correct UI element
  - Whether it reverses the effect of the current action

---

## Input:
- **Current Action**: {action_desc}
- **Choices**: A–F revert options {undo_options}

---

## Output Format:
Directly output the option letter only
"""

def process_action_desc(action_desc):
    if "click" in action_desc or "CLICK" in action_desc or "LONG_PRESS" in action_desc or "long_press" in action_desc:
        result = "Click on the screen. (In the first screenshot, the click point is marked with a filled, semi-transparent red circle.)"
    elif 'type' in action_desc or "TYPE" in action_desc:
        result = action_desc
    elif 'Scroll' in action_desc or 'scroll' in action_desc:
        result = 'Scroll'
    elif " to " in action_desc:
        result = action_desc.split(" to ")[0]
    return result

def parse_args():
    parser = argparse.ArgumentParser(description="InternVL-eval")
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--image_root_odyssey', type=str)
    parser.add_argument('--image_root_ac', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--out_dir', type=str, default='./result/action_reversal')
    parser.add_argument('--result_filename', type=str, default='')
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--max_num', type=int, default=12)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    random.seed(args.seed)

    jsonl_data = read_jsonl_file(args.data_path)

    model_path = args.checkpoint
    image_root_odyssey = args.image_root_odyssey
    image_root_ac = args.image_root_ac

    all_data = jsonl_data
    for i, data in enumerate(all_data):
        data['index'] = i

    if args.result_filename == '':
        result_filename = model_path.rstrip(os.sep).split(os.sep)[-1] + '.jsonl'
    else:
        result_filename = args.result_filename

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
        images = data_item['images']

        action_desc =  process_action_desc(data_item['action_desc'])
        options = data_item['options']

        image_root = image_root_ac if data_item['data_source'] == 'AndroidControl' else image_root_odyssey
        images = [open_image(os.path.join(image_root, img)) for img in images]

        if 'click_point' in data_item:
            images[0] = apply_click_to_image_pil(images[0], *data_item['click_point'])

        question = prompt_format.format(action_desc=action_desc, undo_options=options)
        response = model(question, images, generation_config)

        pred_data = data_item
        pred_data['pred_response'] = response
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
        write_jsonl_file(merged_outputs, result_path)

        acc_list = []

        for data in merged_outputs:
            gt = data['answer']
            pred_response = data['pred_response'].strip()[0]

            acc_list.append(gt.lower() == pred_response.lower())
        print('Accuracy', np.mean(acc_list))

if __name__ == '__main__':
    main()