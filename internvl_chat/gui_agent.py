from PIL import Image, ImageDraw
from collections import deque
import re

from eval.infer import Model

SYSTEM_PROMPT = "You are an intelligent and helpful GUI visual agent, well trained to operate the mobile device UI interface. The valid action space is:\nCLICK[[x, y]]. Click the screen at position [x,y].\nLONG_PRESS[[x, y]]. Long press the screen at position [x, y].\nSCROLL[[x1, y1, x2, y2]]. Scroll from the position [x1, y1] to [x2, y2].\nTYPE[text]. Type in the text.\nMEMORIZE[summary: text; content: text]. Store information into the memory.\nANSWER[text]. Answer with the text.\nOPEN_APP[app_name]. Open the app named app_name.\nPRESS_HOME. Go back to the home screen.\nPRESS_BACK. Go back to the previous screen.\nPRESS_ENTER. Press the enter key.\nWAIT. Wait for device response.\nTASK_COMPLETE. Indicate the task is completed.\nTASK_IMPOSSIBLE. Indicate the task is impossible."

QUESTION_TEMPLATE = "<image>\nThe image is the current screenshot.\n<INSTRUCTION> (user instruction): {instruction}\n<MEMORY> (stored memory content): {memory}\n<PAST ACTIONS> (past actions): {action_pair_history}\nBased on the above information, your task is to reason about the next action and provide your thinking process and the next action. Your output should follow the following format:\n<THOUGHT>: the thinking process\n<ACTION DESC>: the description about the next action\n<ACTION>: the next action"

QUESTION_TEMPLATE_TEMPORAL = "{image_tags}{past_image_desc}\n<image>\nThe image is the current screenshot.\n<INSTRUCTION> (user instruction): {instruction}\n<MEMORY> (stored memory content): {memory}\n<PAST ACTIONS> (past actions): {action_pair_history}\nBased on the above information, your task is to reason about the next action and provide your thinking process and the next action. Your output should follow the following format:\n<THOUGHT>: the thinking process\n<ACTION DESC>: the description about the next action\n<ACTION>: the next action"

def construct_action_history(action_history, action_desc_history):
    action_pairs = []
    for action, action_desc in zip(action_history, action_desc_history):
        action_pairs.append('({}; {})'.format(action_desc, action))
    return '[{}]'.format(', '.join(action_pairs))

def apply_click_to_image_pil(image, x, y, transparency=0.5):
    # Convert the image to RGBA for transparency handling
    image = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))  # Transparent overlay

    # Calculate the position based on normalized coordinates
    width, height = image.size

    # Draw the circle on the overlay
    draw = ImageDraw.Draw(overlay)
    if max(width, height) < 1000:
        radius = 10
    elif min(width, height) >= 1000:
        radius = 20
    else:
        radius = 15
    color = (255, 0, 0, int(255 * transparency))  # Blue color with alpha channel
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius), fill=color
    )

    # Composite the overlay with the original image
    combined = Image.alpha_composite(image, overlay)

    # Convert back to RGB for the output
    return combined.convert("RGB")

def parse_action_output(output, screen_width, screen_height):
    parsed_action = {"action_type": "INVALID", "parameters": []}
    # Define regular expressions for each action type with their specific parameters format
    patterns = {
        "CLICK": r"CLICK\[\[(\d+), (\d+)\]\]",
        "LONG_PRESS": r"LONG_PRESS\[\[(\d+), (\d+)\]\]",
        "SCROLL": r"SCROLL\[\[(\d+), (\d+), (\d+), (\d+)\]\]",
        "TYPE": r"TYPE\[(.+?)\]",
        "ANSWER": r"ANSWER\[(.+?)\]",
        "MEMORIZE": r"MEMORIZE\[(.+?)\]",
        "PRESS_HOME": r"PRESS_HOME",
        "PRESS_BACK": r"PRESS_BACK",
        "PRESS_ENTER": r"PRESS_ENTER",
        "TASK_COMPLETE": r"TASK_COMPLETE",
        "TASK_IMPOSSIBLE": r"TASK_IMPOSSIBLE",
        "WAIT": r"WAIT"
    }
    # Loop over each pattern to find matches
    for action, pattern in patterns.items():
        for match in re.finditer(pattern, output):
            if action == "CLICK":
                x, y = match.groups()
                parameters = [int(int(x)/1000*screen_width), int(int(y)/1000*screen_height)]
            elif action == "LONG_PRESS":
                x, y = match.groups()
                parameters = [int(int(x)/1000*screen_width), int(int(y)/1000*screen_height)]
            elif action == "SCROLL":
                x1, y1, x2, y2 = match.groups()
                parameters = [int(int(x1)/1000*screen_width), int(int(y1)/1000*screen_height), int(int(x2)/1000*screen_width), int(int(y2)/1000*screen_height)]
            elif action == "TYPE":
                text = match.group(1)
                parameters = [text]
            elif action == "MEMORIZE":
                text = match.group(1)
                parameters = [text]
            elif action in ["PRESS_HOME", "PRESS_BACK", "PRESS_ENTER", "TASK_COMPLETE", "TASK_IMPOSSIBLE", "WAIT"]:
                parameters = []
            elif action == "ANSWER":
                text = match.group(1)
                parameters = [text]
            elif action == "OPEN_APP":
                text = match.group(1)
                parameters = [text]
            parsed_action = {"action_type": action, "parameters": parameters}
    return parsed_action

class GUI_Reflection_Agent:
  def __init__(self, model_path, temporal_len=4):
    self._actions = []
    self._action_desc = []
    self.memory = ''
    self.temporal_len = temporal_len
    if temporal_len > 0:
        self.history_images = deque(maxlen=temporal_len)
        self.history_images_annos = deque(maxlen=temporal_len)

    self.model = Model(model_path, auto=False)
    self.generation_config = {
        "max_new_tokens": 1024,
        "do_sample": False,
        "temperature": 0.0,
        "top_k": 0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "num_beams": 1,
    }

  def reset(self) -> None:
    self._actions.clear()
    self._action_desc.clear()
    self.memory = ''
    if self.temporal_len > 0:
        self.history_images = deque(maxlen=self.temporal_len)
        self.history_images_annos = deque(maxlen=self.temporal_len)

  def step(
      self, image: Image, goal: str
  ):
    action_pair_history = construct_action_history(self._actions, self._action_desc)
    if self.temporal_len == 0 or len(self.history_images) == 0:
        question = QUESTION_TEMPLATE.format(instruction=goal, memory=self.memory, action_pair_history=action_pair_history)
        input_images = [image]
    else:
        image_tags = "<image>\n"*len(self.history_images)
        if len(self.history_images) == 1:
            past_image_desc = 'The image is the screenshot from the last step.'
        else:
            past_image_desc = f'The images are the screenshots from the past {len(self.history_images)} steps.'
        question = QUESTION_TEMPLATE_TEMPORAL.format(image_tags=image_tags, past_image_desc=past_image_desc, instruction=goal, memory=self.memory, action_pair_history=action_pair_history)
        input_images = []
        for history_image, history_image_anno in zip(list(self.history_images), list(self.history_images_annos)):
            if history_image_anno is not None:
                history_image = apply_click_to_image_pil(history_image, *history_image_anno)
            history_image = history_image.resize((448, 448))
            input_images.append(history_image)
        input_images.append(image)

    response = self.model(question, input_images, self.generation_config, system_message=SYSTEM_PROMPT)

    action = parse_action_output(response, image.size[0], image.size[1])
    action_desc = response.split('<ACTION DESC>:')[-1].split('<ACTION>:')[0].strip()
    if action['action_type'] == 'MEMORIZE':
        memory_content = action['parameters'][0]
        if len(self.memory):
            self.memory = self.memory.rstrip(']') + ", '{}']".format(memory_content)
        else:
            self.memory = "['{}']".format(memory_content)
        action_brief = action_desc.lstrip('Memorize').strip()
        action_brief = "MEMORIZE[summary: {}]".format(action_brief)
        self._actions.append(action_brief)
        self._action_desc.append(action_desc)
    else:    
        self._actions.append(response.split('<ACTION>:')[-1].strip())
        self._action_desc.append(action_desc)
    if self.temporal_len > 0:
        self.history_images.append(image)
        self.history_images_annos.append(None if action['action_type'] not in ['CLICK', 'LONG_PRESS'] else action['parameters'])

    return action