import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import numpy as np
from PIL import Image
import io
import math

###constants

IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
QUAD_START_TOKEN = "<quad>"
QUAD_END_TOKEN = "</quad>"
REF_START_TOKEN = "<ref>"
REF_END_TOKEN = "</ref>"
BOX_START_TOKEN = "<box>"
BOX_END_TOKEN = "</box>"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.4814546, 0.4578275, 0.40821073)
CLIP_STD = (0.2686295, 0.2613025, 0.2757711)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)


qualities = list(range(75, 101))


def simulate_jpeg_degradation(quality):
    def jpeg_degrade(img):
        with io.BytesIO() as output:
            img.convert("RGB").save(output, format="JPEG", quality=quality)
            output.seek(0)  # Move the reading cursor to the start of the stream
            img_jpeg = Image.open(
                output
            ).copy()  # Use .copy() to make sure the image is loaded in memory
        return img_jpeg

    return jpeg_degrade


jpeg_degrade_functions = {
    quality: simulate_jpeg_degradation(quality) for quality in qualities
}


class ImageProcessor:
    def __init__(self, pad2square=False, normalize_type="imagenet"):
        self.pad2square = pad2square
        self.normalize_type = normalize_type

    def find_closest_aspect_ratio(
        self, aspect_ratio, target_ratios, width, height, image_size
    ):
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def build_transform(self, is_train, image_size):
        if self.normalize_type == "imagenet":
            MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        elif self.normalize_type == "clip":
            MEAN, STD = CLIP_MEAN, CLIP_STD
        elif self.normalize_type == "siglip":
            MEAN, STD = SIGLIP_MEAN, SIGLIP_STD
        else:
            raise NotImplementedError

        if is_train:
            transform = T.Compose(
                [
                    T.Lambda(
                        lambda img: img.convert("RGB") if img.mode != "RGB" else img
                    ),
                    T.RandomChoice(
                        [
                            T.Lambda(jpeg_degrade_functions[quality])
                            for quality in qualities
                        ]
                    ),
                    T.Resize(
                        (image_size, image_size),
                        interpolation=InterpolationMode.BICUBIC,
                    ),
                    T.ToTensor(),
                    T.Normalize(mean=MEAN, std=STD),
                ]
            )
        else:
            if not self.pad2square:
                transform = T.Compose(
                    [
                        T.Lambda(
                            lambda img: img.convert("RGB") if img.mode != "RGB" else img
                        ),
                        T.Resize(
                            (image_size, image_size),
                            interpolation=InterpolationMode.BICUBIC,
                        ),
                        T.ToTensor(),
                        T.Normalize(mean=MEAN, std=STD),
                    ]
                )
            else:
                transform = T.Compose(
                    [
                        T.Lambda(
                            lambda img: img.convert("RGB") if img.mode != "RGB" else img
                        ),
                        T.Lambda(
                            lambda img: self.expand2square(
                                img, tuple(int(x * 255) for x in MEAN)
                            )
                        ),
                        T.Resize(
                            (image_size, image_size),
                            interpolation=InterpolationMode.BICUBIC,
                        ),
                        T.ToTensor(),
                        T.Normalize(mean=MEAN, std=STD),
                    ]
                )
        return transform

    def dynamic_preprocess(
        self, image, image_size, min_num=1, max_num=6, use_thumbnail=False
    ):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        target_ratios = set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def dynamic_preprocess_v2(
        self, image, image_size, min_num=1, max_num=6, use_thumbnail=False
    ):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        max_patch = np.ceil(orig_width * orig_height * 2 / image_size / image_size)

        target_ratios = set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num and i * j <= max_patch
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images
    
    def find_closest_aspect_ratio_v3(self, image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    # llava like, from https://huggingface.co/openbmb/MiniCPM-V-2_6/blob/3f7a8da1b7a8b928b5ee229fae33cf43fd64cf31/image_processing_minicpmv.py#L257 with modification 
        assert min_num == 1
        original_width, original_height = image.size
        log_ratio = math.log(original_width / original_height)
        ratio = original_width * original_height / (image_size * image_size)
        multiple = min(math.ceil(ratio), max_num)
        if multiple <= 1:
            return [1, 1]
        candidate_split_grids_nums = []
        for i in [multiple - 1, multiple, multiple + 1]:
            if i > max_num:
                continue
            candidate_split_grids_nums.append(i)
        
        candidate_grids = []
        for split_grids_nums in candidate_split_grids_nums:
            m = 1
            while m <= split_grids_nums:
                if split_grids_nums % m == 0:
                    candidate_grids.append([m, split_grids_nums // m])
                m += 1
        best_grid = [1, 1]
        min_error = float("inf")
        for grid in candidate_grids:
            error = abs(log_ratio - math.log(grid[0] / grid[1]))
            if error < min_error:
                best_grid = grid
                min_error = error

        return best_grid

    def dynamic_preprocess_v3(self, image, image_size=448, min_num=1, max_num=6,  use_thumbnail=False):
        target_aspect_ratio = self.find_closest_aspect_ratio_v3(image, min_num, max_num, image_size, use_thumbnail)
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images
    
    

    @staticmethod
    def expand2square(pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result
