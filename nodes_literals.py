from sys import float_info
from typing import Any
from nodes import MAX_RESOLUTION
import torch

import numpy as np
from PIL import Image, ImageDraw
import random
import math

class SeedGenerator:
    RETURN_TYPES = ("INT",)
    OUTPUT_TOOLTIPS = ("seed (INT)",)
    FUNCTION = "get_seed"

    CATEGORY = "ImageSaver/utils"
    DESCRIPTION = "Provides seed as integer"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "The random seed used for creating the noise."}),
                "increment": ("INT", {"default": 0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "tooltip": "number to add to the final seed value"}),
            }
        }

    def get_seed(self, seed: int, increment: int) -> tuple[int,]:
        return (seed + increment,)

class StringLiteral:
    RETURN_TYPES = ("STRING",)
    OUTPUT_TOOLTIPS = ("string (STRING)",)
    FUNCTION = "get_string"

    CATEGORY = "ImageSaver/utils"
    DESCRIPTION = "Provides a string"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "string": ("STRING", {"default": "", "multiline": True, "tooltip": "string"}),
            }
        }

    def get_string(self, string: str) -> tuple[str,] :
        return (string,)

class SizeLiteral:
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("size",)
    OUTPUT_TOOLTIPS = ("size (INT)",)
    FUNCTION = "get_int"

    CATEGORY = "ImageSaver/utils"
    DESCRIPTION = f"Provides integer number between 0 and {MAX_RESOLUTION} (step=8)"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "size": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, "tooltip": "size as integer (in steps of 8)"}),
            }
        }

    def get_int(self, size: int) -> tuple[int,]:
        return (size,)

class IntLiteral:
    RETURN_TYPES = ("INT",)
    OUTPUT_TOOLTIPS = ("int (INT)",)
    FUNCTION = "get_int"

    CATEGORY = "ImageSaver/utils"
    DESCRIPTION = "Provides integer number between 0 and 1000000"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "int": ("INT", {"default": 0, "min": 0, "max": 1000000, "tooltip": "integer number"}),
            }
        }

    def get_int(self, int: int) -> tuple[int,]:
        return (int,)

class FloatLiteral:
    RETURN_TYPES = ("FLOAT",)
    OUTPUT_TOOLTIPS = ("float (FLOAT)",)
    FUNCTION = "get_float"

    CATEGORY = "ImageSaver/utils"
    DESCRIPTION = f"Provides a floating point number between {float_info.min} and {float_info.max} (step=0.01)"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "float": ("FLOAT", {"default": 1.0, "min": float_info.min, "max": float_info.max, "step": 0.01, "tooltip": "floating point number"}),
            }
        }

    def get_float(self, float: float):
        return (float,)

class CfgLiteral:
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("value",)
    OUTPUT_TOOLTIPS = ("cfg (FLOAT)",)
    FUNCTION = "get_float"

    CATEGORY = "ImageSaver/utils"
    DESCRIPTION = "Provides CFG value between 0.0 and 100.0"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "tooltip": "CFG as a floating point number"}),
            }
        }

    def get_float(self, cfg: float) -> tuple[float,]:
        return (cfg,)

class ConditioningConcatOptional:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "conditioning_to": ("CONDITIONING", {"tooltip": "base conditioning to concat to (or pass through, if second is empty)"}),
            },
            "optional": {
                "conditioning_from": ("CONDITIONING", {"tooltip": "conditioning to concat to conditioning_to, if empty, then conditioning_to is passed through unchanged"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "concat"
    CATEGORY = "conditioning"

    def concat(self, conditioning_to, conditioning_from=None):
        if conditioning_from is None:
            return (conditioning_to,)

        out = []
        if len(conditioning_from) > 1:
            print("Warning: ConditioningConcat conditioning_from contains more than 1 cond, only the first one will actually be applied to conditioning_to.")

        cond_from = conditioning_from[0][0]
        for i in range(len(conditioning_to)):
            t1 = conditioning_to[i][0]
            tw = torch.cat((t1, cond_from), 1)
            n = [tw, conditioning_to[i][1].copy()]
            out.append(n)

        return (out,)

class RandomShapeGenerator:
    """
    A ComfyUI node that generates images with random shapes.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "width": ("INT", { "default": 512, "min": 64, "max": 4096, "step": 64, "tooltip": "Width of the generated image in pixels" }),
                "height": ("INT", { "default": 512, "min": 64, "max": 4096, "step": 64, "tooltip": "Height of the generated image in pixels" }),
                "bg_color": (["random", "white", "black", "red", "green", "blue", "yellow", "cyan", "magenta"], { "tooltip": "Background color preset or random" }),
                "fg_color": (["random", "black", "white", "red", "green", "blue", "yellow", "cyan", "magenta"], { "tooltip": "Foreground shape color preset or random" }),
                "shape_type": (["random", "circle", "oval", "triangle", "square", "rectangle", "rhombus", "pentagon", "hexagon"], { "tooltip": "Type of shape to generate or random" }),
                "seed": ("INT", { "default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "Random seed for reproducible shape generation" }),
            },
            "optional": {
                "bg_color_override": ("STRING", { "default": "", "multiline": False, "tooltip": "Override background color with hex (#AABBCC) or RGB(r, g, b) format" }),
                "fg_color_override": ("STRING", { "default": "", "multiline": False, "tooltip": "Override foreground color with hex (#AABBCC) or RGB(r, g, b) format" }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "bg_rgb", "fg_rgb")
    OUTPUT_TOOLTIPS = ("Generated image with random shape", "Background color as RGB/hex", "Foreground color as RGB/hex")
    FUNCTION = "generate_shape"
    CATEGORY = "image/generators"
    DESCRIPTION = "Generates images with random shapes for testing and prototyping"

    def __init__(self):
        self.color_map = {
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "yellow": (255, 255, 0),
            "cyan": (0, 255, 255),
            "magenta": (255, 0, 255),
        }

    def parse_rgb_string(self, rgb_str: str) -> tuple[int, int, int] | None:
        """Parse RGB string like 'RGB(123, 45, 67)' or '#AABBCC' into tuple (123, 45, 67)"""
        if not rgb_str or rgb_str.strip() == "":
            return None

        rgb_str = rgb_str.strip()

        try:
            # Try hex format first (#AABBCC or AABBCC)
            if rgb_str.startswith("#"):
                hex_str = rgb_str[1:]
            else:
                hex_str = rgb_str

            # Check if it's a valid hex string (6 characters)
            if len(hex_str) == 6 and all(c in '0123456789ABCDEFabcdef' for c in hex_str):
                r = int(hex_str[0:2], 16)
                g = int(hex_str[2:4], 16)
                b = int(hex_str[4:6], 16)
                return (r, g, b)

            # Try RGB(r, g, b) format
            rgb_str_upper = rgb_str.upper()
            if rgb_str_upper.startswith("RGB(") and rgb_str_upper.endswith(")"):
                values = rgb_str[4:-1].split(",")
                r, g, b = [int(v.strip()) for v in values]
                # Validate range
                if all(0 <= val <= 255 for val in [r, g, b]):
                    return (r, g, b)
        except (ValueError, IndexError):
            return None

        return None

    def draw_shape(self, draw: ImageDraw.ImageDraw, img_width: int, img_height: int, shape_type: str, shape_color: tuple[int, int, int]) -> None:
        """Draw a random shape on the image."""

        # Random size - prefer larger sizes (40-70% of image dimensions)
        size_factor = random.uniform(0.4, 0.7)
        shape_width = int(img_width * size_factor)
        shape_height = int(img_height * size_factor)

        # Random position (ensure shape stays fully within bounds)
        x = random.randint(0, max(0, img_width - shape_width))
        y = random.randint(0, max(0, img_height - shape_height))

        # Draw the shape based on type
        if shape_type == 'circle':
            # Make it a perfect circle using the minimum dimension
            radius = min(shape_width, shape_height) // 2
            draw.ellipse([x, y, x + radius * 2, y + radius * 2], fill=shape_color)

        elif shape_type == 'oval':
            draw.ellipse([x, y, x + shape_width, y + shape_height], fill=shape_color)

        elif shape_type == 'square':
            # Make it a perfect square
            side = min(shape_width, shape_height)
            draw.rectangle([x, y, x + side, y + side], fill=shape_color)

        elif shape_type == 'rectangle':
            draw.rectangle([x, y, x + shape_width, y + shape_height], fill=shape_color)

        elif shape_type == 'triangle':
            # Equilateral-ish triangle
            points = [
                (x + shape_width // 2, y),  # top
                (x, y + shape_height),  # bottom left
                (x + shape_width, y + shape_height)  # bottom right
            ]
            draw.polygon(points, fill=shape_color)

        elif shape_type == 'rhombus':
            # Diamond shape
            points = [
                (x + shape_width // 2, y),  # top
                (x + shape_width, y + shape_height // 2),  # right
                (x + shape_width // 2, y + shape_height),  # bottom
                (x, y + shape_height // 2)  # left
            ]
            draw.polygon(points, fill=shape_color)

        elif shape_type == 'pentagon':
            # Regular pentagon
            cx, cy = x + shape_width // 2, y + shape_height // 2
            radius = min(shape_width, shape_height) // 2
            points = []
            for i in range(5):
                angle = i * 2 * math.pi / 5 - math.pi / 2
                px = cx + radius * math.cos(angle)
                py = cy + radius * math.sin(angle)
                points.append((px, py))
            draw.polygon(points, fill=shape_color)

        elif shape_type == 'hexagon':
            # Regular hexagon
            cx, cy = x + shape_width // 2, y + shape_height // 2
            radius = min(shape_width, shape_height) // 2
            points = []
            for i in range(6):
                angle = i * 2 * math.pi / 6
                px = cx + radius * math.cos(angle)
                py = cy + radius * math.sin(angle)
                points.append((px, py))
            draw.polygon(points, fill=shape_color)

    def generate_shape(self, width: int, height: int, bg_color: str, fg_color: str, shape_type: str, seed: int, bg_color_override: str = "", fg_color_override: str = "") -> tuple[torch.Tensor, str, str]:
        """Generate an image with a random shape."""

        # Set random seed for reproducibility
        random.seed(seed)

        # Get colors from map or generate random RGB values
        # Check for override first
        bg_override = self.parse_rgb_string(bg_color_override)
        if bg_override is not None:
            bg_rgb = bg_override
        elif bg_color == "random":
            bg_rgb = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        else:
            bg_rgb = self.color_map.get(bg_color, (255, 255, 255))

        fg_override = self.parse_rgb_string(fg_color_override)
        if fg_override is not None:
            fg_rgb = fg_override
        elif fg_color == "random":
            fg_rgb = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        else:
            fg_rgb = self.color_map.get(fg_color, (0, 0, 0))

        # Create image
        img = Image.new('RGB', (width, height), bg_rgb)
        draw = ImageDraw.Draw(img)

        # Select shape type
        if shape_type == "random":
            shapes = ['circle', 'oval', 'triangle', 'square', 'rectangle', 'rhombus', 'pentagon', 'hexagon']
            selected_shape = random.choice(shapes)
        else:
            selected_shape = shape_type

        # Draw the shape
        self.draw_shape(draw, width, height, selected_shape, fg_rgb)

        # Convert PIL Image to torch tensor (ComfyUI format)
        # ComfyUI expects images in format [batch, height, width, channels] with values 0-1
        img_array = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array)[None,]

        # Format RGB values as strings for output (both formats)
        bg_hex = f"#{bg_rgb[0]:02X}{bg_rgb[1]:02X}{bg_rgb[2]:02X}"
        fg_hex = f"#{fg_rgb[0]:02X}{fg_rgb[1]:02X}{fg_rgb[2]:02X}"
        bg_rgb_str = f"RGB({bg_rgb[0]}, {bg_rgb[1]}, {bg_rgb[2]}) / {bg_hex}"
        fg_rgb_str = f"RGB({fg_rgb[0]}, {fg_rgb[1]}, {fg_rgb[2]}) / {fg_hex}"

        return (img_tensor, bg_rgb_str, fg_rgb_str)

