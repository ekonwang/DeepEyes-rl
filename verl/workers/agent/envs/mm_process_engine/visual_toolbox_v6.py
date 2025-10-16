import numpy as np
import copy
import math
from verl.workers.agent.tool_envs import ToolBase
from typing import Optional, List, Dict, Any
from PIL import Image
import re
import json
from verl.workers.agent.envs.mm_process_engine.prompt import PROMPT
from math import ceil, floor
# 临时修复
# ToolBase.registry = {}

def compute_crop_scaled_size(resize_h, resize_w, crop_h, crop_w, max_len=32768, add_len=4000):
    """
    计算crop图像的缩放因子，保证总的prompt length不超过max_len。
    保持长宽比，不放大，只缩小。
    """
    A = (resize_h * resize_w) / (28 * 28) + add_len
    B = (crop_h * crop_w) / (28 * 28)

    if B <= 0:
        return 1.0  # 没有crop图，scale=1

    max_sqr = (max_len - A) / B
    if max_sqr <= 0:
        return 0.0  # 已经超限，无法再加crop图

    s = math.sqrt(max_sqr)
    s = min(s, 1.0)

    scaled_h = int(crop_h * s)
    scaled_w = int(crop_w * s)

    # 向下取整到 28 的倍数
    scaled_h = max(28, (scaled_h // 28) * 28)
    scaled_w = max(28, (scaled_w // 28) * 28)

    return scaled_h, scaled_w

class VisualToolBoxV6(ToolBase):
    name = "visual_toolbox_v6"
    # user_prompt = "Here is the cropped image returned after you calling the function {}.\nIf the images provided above are sufficient to answer the user's question, please put your final answer within <answer></answer>. Otherwise you can continue to call tools within <tool_call></tool_call>."

    user_prompt = PROMPT.USER_PROMPT_V2
    def __init__(self, _name, _desc, _params, **kwargs):
        super().__init__(
            name=self.name,
        )
        self.chatml_history = []
        self.multi_modal_data = None  # To store the current image being processed


    def extract_answer(self, action_string: str) -> Dict[str, any]:
        answer = re.findall(r'<answer>(.*?)</answer>', action_string, re.DOTALL)
        return answer[-1] if answer else None
        
    def extract_action(self, action_string: str) -> Dict[str, Any]:
        """
        Extracts the tool call from the action string.
        
        Args:
            action_string: The string containing the tool call in XML tags.
            
        Returns:
            A dictionary with the tool name and arguments.
            
        Raises:
            ValueError: If no tool call is found or JSON is invalid.
        """
        tool_call_match = re.findall(r'<tool_call>(.*?)</tool_call>', action_string, re.DOTALL)
        return tool_call_match[-1] if tool_call_match else None

    def execute(self, action_string, vllm_input_list, **kwargs) -> tuple:
        """
        Execute the tool functionality based on the action string.
        
        Args:
            action_string: The string containing the tool call in XML tags.
            
        Returns:
            observation: The structured observation with the processed image.
            reward: 0.1 if tool call is successful with correct JSON format, 0 otherwise.
            done: Whether the episode is terminated.
            info: Additional info.
        """
        
        answer = self.extract_answer(action_string)
        if answer:
            return "", 0.0, True, {}
        action = self.extract_action(action_string)
        if not action:
            return "", 0.0, True, {}
        try:
            tool_call = json.loads(action.strip())  # 或使用 literal_eval
        except Exception as e:
            error_msg = f"Invalid tool call format: {action.strip()}. Error: {e}"
            obs = "\n<|im_start|>user\n" + f"Error: {str(error_msg)}" + "<|im_end|>\n<|im_start|>assistant\n"
            info = {"error": str(e), "status": "failed"}
            return obs, 0.0, False, {}
        try:

            tool_name = tool_call["name"]
            args = tool_call["arguments"]
        
            if tool_name == "image_zoom_in_tool":
                # Zoom in by cropping the image
                # image_path = args["image_path"]
                bbox = args["bbox_2d"]
                if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > 1 or bbox[3] > 1:
                    raise ValueError(f"ZOOM IN BBOX SHOULD BE IN [0,1] RANGE {bbox=}")
                if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                    raise ValueError(f"ZOOM IN BBOX IS INVALID {bbox=}")
                image_index = args["index"]
                if not bbox:
                    raise ValueError(f"ZOOM IN ARGUMENTS ARE INVALID")
                # img = Image.open(image_path)
                img = vllm_input_list['multi_modal_data']['image'][image_index]
                w, h = img.size
                bbox_absolute = [
                    max(0, int(math.floor(bbox[0] * w))),
                    max(0, int(math.floor(bbox[1] * h))),
                    min(int(math.ceil(bbox[2] * w)), w),
                    min(int(math.ceil(bbox[3] * h)), h),
                ]
                if bbox_absolute[0] >= bbox_absolute[2] or bbox_absolute[1] >= bbox_absolute[3]:
                    raise ValueError(f"ZOOM IN BBOX IS INVALID {bbox=}, {bbox_absolute=}, {w=}, {h=}")
                if (bbox_absolute[2] - bbox_absolute[0]) / (bbox_absolute[3] - bbox_absolute[1]) > 200 or (bbox_absolute[3] - bbox_absolute[1]) / (bbox_absolute[2] - bbox_absolute[0]) > 200:
                    raise ValueError(f"ZOOM IN BBOX APSECT RATIO IS TOO LARGE {bbox=}, {bbox_absolute=}, {w=}, {h=}")
                cropped_img = img.crop(bbox_absolute)
                current_image = cropped_img

                # # zoom in
                # w, h = img.size
                # crop_w, crop_h = cropped_img.size
                # shorter_side = min(w, h)
                # longer_crop_side = max(crop_w, crop_h)
                # zoom_factor = max(1, (shorter_side / longer_crop_side * 0.8))
                # new_w = int(crop_w * zoom_factor)
                # new_h = int(crop_h * zoom_factor)
                # new_safe_h, new_safe_w = compute_crop_scaled_size(h, w, new_h, new_w)
                # if new_safe_h != new_h or new_safe_w != new_w:
                #     print(
                #         f"[DEBUG] Crop Image Size Adjusted: "
                #         f"from ({new_h, new_w}) "
                #         f"to ({new_safe_h, new_safe_w}) "
                #         f"with scale ({new_safe_h / new_h:.2f}, "
                #         f"{new_safe_w / new_w:.2f})"
                #     )

                # zoomed_img = cropped_img.resize((new_safe_w, new_safe_h), Image.BICUBIC)
                # current_image = zoomed_img
            
            elif tool_name == "image_rotate_tool":
                # Rotate the image
                # image_path = args["image_path"]
                angle = args["angle"]
                # img = Image.open(image_path)
                img = self.multi_modal_data['image'][0]
                rotated_img = img.rotate(angle)
                current_image = rotated_img
                
            else:
                raise ValueError(f"Unknown tool name: {tool_name}")
            # Prepare the observation
            obs = {
                "prompt": "\n<|im_start|>user\n" + "<tool_response>" +"<image>" + self.user_prompt + "</tool_response>" + "<|im_end|>\n<|im_start|>assistant\n",
                "multi_modal_data": {"image": [current_image]}
            }
            reward = 0.0  # Reward for successful tool call with correct JSON
            done = False
            info = {"status": "success", "tool_used": tool_name}
            print(f'[DEBUG-visual_toolbox_v2] SUCCESS ACTION {action_string=}')
            return obs, reward, done, info
        except Exception as e:
            # Return an error observation if something goes wrong
            print(f'[DEBUG-visual_toolbox_v2] Execute WRONG - {str(e)} {action_string=}')
            obs = "\n<|im_start|>user\n" + f"Error: {str(e)}" + "<|im_end|>\n<|im_start|>assistant\n"
            reward = 0.0  # No reward for failed execution
            done = False
            info = {"error": str(e), "status": "failed"}
            return obs, reward, done, info

    def reset(self, raw_prompt, multi_modal_data, origin_multi_modal_data, **kwargs):
        self.chatml_history = raw_prompt
        self.multi_modal_data = origin_multi_modal_data
        assert 'image' in self.multi_modal_data.keys(), f'[ERROR] {origin_multi_modal_data=}'
        assert len(self.multi_modal_data['image']) > 0, f'[ERROR] {self.multi_modal_data["image"]=}'
        
        self.height = self.multi_modal_data['image'][0].height
        self.width = self.multi_modal_data['image'][0].width

    def validate_bbox(self, left, top, right, bottom):
        try:
            assert left < right and bottom > top, f'invalid shape for {left=}, {top=}, {right=}, {bottom=}'
            height = bottom - top
            width = right - left
            assert max(height, width) / min(height, width) <= 100, f"aspect ratio error: {left=}, {top=}, {right=}, {bottom=}"
            assert min(height, width) > 30, f"{height=}, {width=} is too small"
            return True
        except Exception as err:
            print(f' [ERROR vl_agent #2] {err=}')
            return False


    def maybe_resize_bbox(self, left, top, right, bottom):
        left = max(0, left)
        top = max(0, top)
        right = min(self.width, right)
        bottom = min(self.height, bottom)
        if not self.validate_bbox(left, top, right, bottom):
            return None

        height = bottom - top
        width = right - left
        if height < 28 or width < 28:
            center_x = (left + right) / 2.0
            center_y = (top + bottom) / 2.0
            ratio = 28 / min(height, width)
            new_half_height = ceil(height * ratio * 0.5)
            new_half_width = ceil(width * ratio * 0.5)
            new_left = floor(center_x - new_half_width)
            new_right = ceil(center_x + new_half_width)
            new_top = floor(center_y - new_half_height)
            new_bottom = ceil(center_y + new_half_height)
            if not self.validate_bbox(new_left, new_top, new_right, new_bottom):
                return None
            return [new_left, new_top, new_right, new_bottom]
        return [left, top, right, bottom]


if __name__ == "__main__":
    # Example usage (for testing)
    tool = VisualToolBox("visual_toolbox", "Tool for image processing", {})
    
    # Test zoom in tool (should return reward=0.1)
    zoom_in_action = """
    <tool_call>
    {"name": "image_zoom_in_tool", "arguments": {"image_path": "test.jpg", "bbox": [10, 10, 100, 100]}}
    </tool_call>
    """
    obs, reward, done, info = tool.execute(zoom_in_action)
    print(f"Zoom in result - Reward: {reward}, Info: {info}")
    
    # Test rotate tool (should return reward=0.1)
    rotate_action = """
    <tool_call>
    {"name": "image_rotate_tool", "arguments": {"image_path": "test.jpg", "angle": 90}}
    </tool_call>
    """
    obs, reward, done, info = tool.execute(rotate_action)
    print(f"Rotate result - Reward: {reward}, Info: {info}")
    
    # Test invalid JSON (should return reward=0.0)
    invalid_action = """
    <tool_call>
    {"name": "image_rotate_tool", "arguments": {"image_path": "test.jpg", "angle": 90}
    </tool_call>
    """
    obs, reward, done, info = tool.execute(invalid_action)
    print(f"Invalid JSON result - Reward: {reward}, Info: {info}")
    
    # Test unknown tool (should return reward=0.0)
    unknown_tool_action = """
    <tool_call>
    {"name": "unknown_tool", "arguments": {"param": "value"}}
    </tool_call>
    """
    obs, reward, done, info = tool.execute(unknown_tool_action)
    print(f"Unknown tool result - Reward: {reward}, Info: {info}")
