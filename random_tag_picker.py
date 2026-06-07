import csv
import os
import random


def _process_tag(tag: str, replace_underscore: bool, escape_parens: bool = True) -> str:
    tag = tag.strip()
    if replace_underscore:
        tag = tag.replace("_", " ")
    if escape_parens:
        tag = tag.replace("(", "\\(").replace(")", "\\)")
    return tag


def _split_csv_field(value: str, replace_underscore: bool, escape_parens: bool = True) -> list[str]:
    return [_process_tag(t, replace_underscore, escape_parens) for t in value.split(",") if t.strip()]


class RandomTagPicker:
    """Pick N random tags from a CSV file (first column) and join them with a delimiter."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": "", "multiline": False}),
                "count": ("INT", {"default": 5, "min": 1, "max": 1000, "step": 1}),
                "delimiter": ("STRING", {"default": ", ", "multiline": False}),
                "replace_underscore": ("BOOLEAN", {"default": False}),
                "escape_parens": ("BOOLEAN", {"default": True}),
                "trailing_comma": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tags",)
    FUNCTION = "pick_random_tags"
    CATEGORY = "utils"

    def pick_random_tags(self, file_path: str, count: int, delimiter: str, replace_underscore: bool, escape_parens: bool, trailing_comma: bool, seed: int) -> tuple[str]:
        with open(os.path.expanduser(file_path), newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
            if rows and rows[0][0].strip().lower() == "tag":
                rows = rows[1:]

        tags = [row[0] for row in rows if row and row[0].strip()]
        sample_size = min(count, len(tags))
        rng = random.Random(seed)
        selected = rng.sample(tags, sample_size)
        processed = [_process_tag(t, replace_underscore, escape_parens) for t in selected]
        result = delimiter.join(processed)
        if trailing_comma:
            result += ","
        return (result,)


class RandomCharacterPicker:
    """Pick N random characters from a CSV file and return their trigger tags, with optional core_tags and copyright."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": "", "multiline": False}),
                "count": ("INT", {"default": 1, "min": 1, "max": 1000, "step": 1}),
                "delimiter": ("STRING", {"default": ", ", "multiline": False}),
                "replace_underscore": ("BOOLEAN", {"default": True}),
                "escape_parens": ("BOOLEAN", {"default": True}),
                "trailing_comma": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "include_core_tags": ("BOOLEAN", {"default": False}),
                "include_copyright": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tags",)
    FUNCTION = "pick_random_characters"
    CATEGORY = "utils"

    def pick_random_characters(self, file_path: str, count: int, delimiter: str, replace_underscore: bool, escape_parens: bool, trailing_comma: bool, seed: int, include_core_tags: bool, include_copyright: bool) -> tuple[str]:
        with open(os.path.expanduser(file_path), newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        rng = random.Random(seed)
        selected = rng.sample(rows, min(count, len(rows)))

        parts: list[str] = []
        for row in selected:
            parts.extend(_split_csv_field(row.get("trigger", ""), replace_underscore, escape_parens))
            if include_core_tags:
                parts.extend(_split_csv_field(row.get("core_tags", ""), replace_underscore, escape_parens))
            if include_copyright:
                parts.extend(_split_csv_field(row.get("copyright", ""), replace_underscore, escape_parens))

        result = delimiter.join(parts)
        if trailing_comma:
            result += ","
        return (result,)


class RandomArtistPicker:
    """Pick N random artists from a CSV file and return their trigger values joined with a delimiter."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": "", "multiline": False}),
                "count": ("INT", {"default": 1, "min": 1, "max": 1000, "step": 1}),
                "delimiter": ("STRING", {"default": ", ", "multiline": False}),
                "replace_underscore": ("BOOLEAN", {"default": True}),
                "escape_parens": ("BOOLEAN", {"default": True}),
                "trailing_comma": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tags",)
    FUNCTION = "pick_random_artists"
    CATEGORY = "utils"

    def pick_random_artists(self, file_path: str, count: int, delimiter: str, replace_underscore: bool, escape_parens: bool, trailing_comma: bool, seed: int) -> tuple[str]:
        with open(os.path.expanduser(file_path), newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        rng = random.Random(seed)
        selected = rng.sample(rows, min(count, len(rows)))

        triggers = [_process_tag(row.get("trigger", ""), replace_underscore, escape_parens) for row in selected if row.get("trigger", "").strip()]
        result = delimiter.join(triggers)
        if trailing_comma:
            result += ","
        return (result,)
