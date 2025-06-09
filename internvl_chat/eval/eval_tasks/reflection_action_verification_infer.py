import os
import json
import torch
import random
import argparse
import itertools
from tqdm import tqdm

from eval.infer import Model
from eval.utils import read_jsonl_file, write_jsonl_file, open_image, InferenceSampler

prompt_format = """<image>\n<image>\nGiven an action purpose and two screenshots (the first screenshot corresponds to the step before a certain action while the second one is the outcome screenshot after the execution of the action). You need to judge whether the action purpose has been satisfied by the action executed between these screenshots based on the screenshots content. Directly answer Yes or No. 

## Input
action purpose: {action_purpose}

## Output:
"""

def parse_args():
    parser = argparse.ArgumentParser(description="InternVL-eval")
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--image_root', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--out_dir', type=str, default='./result/action_verification')
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
    image_root = args.image_root

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

        positive_action_purpose = data_item['positive_action_purpose']
        negative_action_purpose = data_item['negative_action_purpose']

        images = [open_image(os.path.join(image_root, img)) for img in images]
        positive_question = prompt_format.format(action_purpose=positive_action_purpose)
        negative_question = prompt_format.format(action_purpose=negative_action_purpose)
        positive_response = model(positive_question, images, generation_config)
        negative_response = model(negative_question, images, generation_config)

        pred_data = data_item
        pred_data['positive_response'] = positive_response
        pred_data['negative_response'] = negative_response
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

        positive_correct = 0
        negative_correct = 0
        for data in merged_outputs:
            positive_response = data['positive_response']
            negative_response = data['negative_response']
            if positive_response.strip().lower()[:3] == 'yes':
                positive_correct += 1
            if negative_response.strip().lower()[:2] == 'no':
                negative_correct += 1
        
        print(positive_correct, positive_correct/len(merged_outputs))
        print(negative_correct, negative_correct/len(merged_outputs))
        print((negative_correct+positive_correct), (negative_correct+positive_correct)/len(merged_outputs)/2)

if __name__ == '__main__':
    main()