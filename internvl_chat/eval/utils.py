import json
import torch
import io
from PIL import Image, ImageDraw

from aoss_client.client import Client
conf_path = '/mnt/afs1/tianhao2/aoss.conf'
client = Client(conf_path)

def read_json_file(file_path):
      with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
      return data
def read_jsonl_file(file_path):
      with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
      return [json.loads(line) for line in lines]
def write_jsonl_file(data, path):
    with open(path, 'w', encoding='utf-8') as result_file:
        for line in data:
            json_line = json.dumps(line, ensure_ascii=False)
            result_file.write(json_line + '\n')
            result_file.flush()

def open_image(image_path):
    if 's3://' in image_path:
        image = Image.open(io.BytesIO(client.get(image_path)))
    else:
        image = Image.open(image_path)
    return image

class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def read_jsonl(path):
    data = []
    with open(path, "r", encoding="utf8") as f_out:
        for line in f_out:
            line = line.strip()
            data.append(json.loads(line))
    return data

def is_valid_bbox(bbox):
    """
    Checks that bbox is a sequence [x1, y1, x2, y2] with 
    0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000.
    
    :param bbox: list or tuple of four numbers
    :return: True if valid, False otherwise
    """
    # must be length 4
    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
        return False

    x1, y1, x2, y2 = bbox

    # all coords must be int/float
    for v in (x1, y1, x2, y2):
        if not isinstance(v, (int, float)):
            return False

    # ordering and bounds check
    if not (0 <= x1 < x2 <= 1000):
        return False
    if not (0 <= y1 < y2 <= 1000):
        return False

    return True

def apply_click_to_image_pil(image, x, y, transparency=0.5):
    # Convert the image to RGBA for transparency handling
    image = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))  # Transparent overlay

    # Calculate the position based on normalized coordinates
    width, height = image.size
    x = int(x/1000 * width)
    y = int(y/1000 * height)

    # Draw the circle on the overlay
    draw = ImageDraw.Draw(overlay)
    if max(width, height) < 1000:
        radius = 15
    elif min(width, height) >= 1000:
        radius = 35
    else:
        radius = 25
    color = (255, 0, 0, int(255 * transparency))  # Blue color with alpha channel
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius), fill=color
    )

    # Composite the overlay with the original image
    combined = Image.alpha_composite(image, overlay)

    # Convert back to RGB for the output
    return combined.convert("RGB")

def draw_bbox_to_image_pil(image, bboxes, color=(255, 0, 0)):
    for bbox in bboxes:
        if not is_valid_bbox(bbox):
            continue
        x1, y1, x2, y2 = bbox
        width = image.width
        height = image.height

        # Convert normalized coordinates to pixel coordinates
        x1 = x1 / 1000 * width
        x2 = x2 / 1000 * width
        y1 = y1 / 1000 * height
        y2 = y2 / 1000 * height

        # Expand the box 3 pixels outward while respecting boundaries
        x1 = max(x1 - 3, 0)
        y1 = max(y1 - 3, 0)
        x2 = min(x2 + 3, width)
        y2 = min(y2 + 3, height)

        # Draw the box on the image
        draw = ImageDraw.Draw(image)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)

    return image