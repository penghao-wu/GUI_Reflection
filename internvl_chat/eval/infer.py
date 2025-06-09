import json
import os
import math
import re
import torch

from transformers import AutoTokenizer, AutoModel, AutoConfig

from eval.preprocess import ImageProcessor
from internvl.conversation import get_conv_template
from internvl.model.internvl_chat import (
    InternVLChatModel,
    InternVLChatConfig,
)

from transformers import CONFIG_MAPPING, MODEL_MAPPING

CONFIG_MAPPING.register("internvl_chat", InternVLChatConfig)
MODEL_MAPPING.register(InternVLChatConfig, InternVLChatModel)

class Model:
    def __init__(self, model_path, max_num=12, auto=True):
        if auto:
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
            config_path = os.path.join(model_path, 'config.json')
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Configuration file not found at {config_path}")           
            with open(config_path, 'r') as config_file:
                config = json.load(config_file)
                name_or_path = config.get('_name_or_path', '').lower()
            if 'internlm7b' in name_or_path:
                device_map = self.split_model('InternVL2-8B')
                kwargs = {'device_map': device_map}
            elif 'internlm20b' in name_or_path:
                device_map = self.split_model('InternVL2-26B')
                kwargs = {'device_map': device_map}
            elif 'qwen2_72b' in name_or_path:
                    device_map = self.split_model('InternVL2-Qwen2-78B')
                    kwargs = {'device_map': device_map}
            elif 'qwen2_5_72b' in name_or_path:
                device_map = self.split_model('InternVL2-Qwen2-78B')
                kwargs = {'device_map': device_map}
            else:
                kwargs = {'device_map': 'auto'}
        else:
            kwargs = {'device_map': 'cuda'}

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=True
        )
        self.preprocessor = ImageProcessor()
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,**kwargs).eval()

        self.image_size = (
            self.model.config.force_image_size
            or self.model.config.vision_config.image_size
        )
        self.use_thumbnail = True
        self.transform = self.preprocessor.build_transform(
            is_train=False, image_size=self.image_size
        )

        self.template_name = "internlm2-chat-v3"

        self.patch_size = config.vision_config.patch_size
        self.num_image_token = int(
            (self.image_size // self.patch_size) ** 2 * (config.downsample_ratio**2)
        )
        self.max_num = max_num
        
    def split_model(self, model_name):
        device_map = {}
        world_size = torch.cuda.device_count()
        num_layers = {'InternVL2-8B': 32, 'InternVL2-26B': 48,
                    'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80, 'InternVL2-Qwen2-78B': 80}[model_name]
        # Since the first GPU will be used for ViT, treat it as half a GPU.
        num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
        num_layers_per_gpu = [num_layers_per_gpu] * world_size
        num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for j in range(num_layer):
                device_map[f'language_model.model.layers.{layer_cnt}'] = i
                layer_cnt += 1
        device_map['vision_model'] = 0
        device_map['mlp1'] = 0
        device_map['language_model.model.tok_embeddings'] = 0
        device_map['language_model.model.embed_tokens'] = 0
        device_map['language_model.output'] = 0
        device_map['language_model.model.norm'] = 0
        device_map['language_model.lm_head'] = 0
        device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

        return device_map

    def __call__(self, question, images, generation_config, system_message=""):
        if '<image>' in question:
            matches = re.findall(r"<image>", question)
            assert len(matches) == len(images), f'<image> error {question} {len(images)}' + question
            question = re.sub(r'(<image>)(?!\n)', r'\1\n', question)
        else:
            question = '<image>\n'*len(images) + question

        template = get_conv_template(self.template_name)
        template.system_message = system_message

        pixel_values = []
        dynamic_nums = []
        if len(images):
            for i, image in enumerate(images):
                processed_images = self.preprocessor.dynamic_preprocess_v3(
                    image,
                    max_num=self.max_num,
                    image_size=self.image_size,
                    use_thumbnail=self.use_thumbnail,
                )
                pixel_value = [
                    self.transform(image) for image in processed_images
                ]
                pixel_values.extend(pixel_value)
                dynamic_nums.append(len(processed_images))

            pixel_values = torch.stack(pixel_values).to(torch.bfloat16).cuda()

            image_tokens_list = list()
            total_dynamic_num = 0
            for index in range(len(dynamic_nums)):
                dynamic_num = dynamic_nums[index]
                total_dynamic_num += dynamic_num
                # print(f'dynamic ViT batch size: {dynamic_num}')
                image_tokens = "<img>" + "<IMG_CONTEXT>" * self.num_image_token * dynamic_num + "<|im_end|>"
                image_tokens_list.append(image_tokens)
            assert total_dynamic_num == pixel_values.shape[0], f'dynamic num not equal, {total_dynamic_num}, {pixel_values.shape[0]}'
            image_tokens_iter = iter(image_tokens_list)

            question = re.sub(r'(<image>)(?!\n)', r'\1\n', question)
            question = re.sub(r"<image>", lambda match:next(image_tokens_iter), question)
        else:
            pixel_values = None
        img_context_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        self.model.img_context_token_id = img_context_token_id
        eos_token_id = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids(["<|im_end|>"])[0],
        ]
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        
        query = template.get_prompt()
        model_inputs = self.tokenizer(
            query, return_tensors="pt", add_special_tokens=False
        )
        input_ids = model_inputs["input_ids"].cuda()
        attention_mask = model_inputs["attention_mask"].cuda()
        generation_config['eos_token_id'] = eos_token_id

        generation_output = self.model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_config,
            )

        outputs = generation_output[0].cpu().tolist()
        response = (
            self.tokenizer.decode(outputs, skip_special_tokens=True)
            .split("assistant\n")[-1]
            .split("<|im_end|>")[0]
            .strip()
        )

        return response