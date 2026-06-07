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


def _get_weights(rows: list[dict]) -> list[float]:
    return [max(float(row.get("count", 1) or 1), 1.0) for row in rows]


def _weighted_sample(rng: random.Random, rows: list, weights: list[float], k: int) -> list:
    rows, weights = list(rows), list(weights)
    selected = []
    for _ in range(min(k, len(rows))):
        [chosen] = rng.choices(rows, weights=weights, k=1)
        idx = rows.index(chosen)
        selected.append(chosen)
        rows.pop(idx)
        weights.pop(idx)
    return selected


def _normalize(value: str) -> str:
    return value.strip().lower().replace("_", " ")


def _parse_exclude(exclude: str) -> set[str]:
    return {_normalize(e) for e in exclude.split(",") if e.strip()}


def _sample(rng: random.Random, rows: list[dict], k: int, weight_by_count: bool) -> list[dict]:
    if weight_by_count:
        return _weighted_sample(rng, rows, _get_weights(rows), k)
    return rng.sample(rows, min(k, len(rows)))


class RandomTagPicker:
    """Pick N random tags from a CSV file (tag column) and join them with a delimiter."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path":          ("STRING",  {"default": "", "multiline": False,                 "tooltip": "path to a CSV file with a 'tag' column"}),
                "count":              ("INT",     {"default": 5, "min": 1, "max": 1000, "step": 1,    "tooltip": "number of tags to pick"}),
                "delimiter":          ("STRING",  {"default": ", ", "multiline": False,               "tooltip": "string used to join picked tags"}),
                "replace_underscore": ("BOOLEAN", {"default": False,                                  "tooltip": "replace underscores with spaces in output"}),
                "escape_parens":      ("BOOLEAN", {"default": True,                                   "tooltip": "escape ( and ) as \\( and \\) to prevent ComfyUI from interpreting them as attention weights"}),
                "trailing_comma":     ("BOOLEAN", {"default": False,                                  "tooltip": "append a trailing comma to the result"}),
                "weight_by_count":    ("BOOLEAN", {"default": False,                                  "tooltip": "use the count column as sampling weight — tags with higher counts are more likely to be picked"}),
                "seed":               ("INT",     {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "random seed — same seed produces the same selection"}),
                "exclude":            ("STRING",  {"default": "", "multiline": False,                 "tooltip": "comma-separated tags to exclude from the pool before sampling (case-insensitive)"}),
                "filter":             ("STRING",  {"default": "", "multiline": False,                 "tooltip": "only consider tags whose name contains this substring (case-insensitive); leave empty for no filtering"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tags",)
    FUNCTION = "pick_random_tags"
    CATEGORY = "utils"

    def pick_random_tags(self, file_path: str, count: int, delimiter: str, replace_underscore: bool, escape_parens: bool, trailing_comma: bool, weight_by_count: bool, seed: int, exclude: str, filter: str) -> tuple[str]:
        excluded = _parse_exclude(exclude)
        needle = _normalize(filter)
        with open(os.path.expanduser(file_path), newline="", encoding="utf-8") as f:
            rows = [row for row in csv.DictReader(f)
                    if row.get("tag", "").strip()
                    and (not needle or needle in _normalize(row["tag"]))
                    and _normalize(row["tag"]) not in excluded]

        rng = random.Random(seed)
        selected = _sample(rng, rows, count, weight_by_count)
        processed = [_process_tag(row["tag"], replace_underscore, escape_parens) for row in selected]
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
                "file_path":          ("STRING",  {"default": "", "multiline": False,                 "tooltip": "path to a CSV file with character, trigger, core_tags, copyright, and count columns"}),
                "count":              ("INT",     {"default": 1, "min": 1, "max": 1000, "step": 1,    "tooltip": "number of characters to pick"}),
                "delimiter":          ("STRING",  {"default": ", ", "multiline": False,               "tooltip": "string used to join all output tags"}),
                "replace_underscore": ("BOOLEAN", {"default": True,                                   "tooltip": "replace underscores with spaces in output"}),
                "escape_parens":      ("BOOLEAN", {"default": True,                                   "tooltip": "escape ( and ) as \\( and \\) to prevent ComfyUI from interpreting them as attention weights"}),
                "trailing_comma":     ("BOOLEAN", {"default": False,                                  "tooltip": "append a trailing comma to the result"}),
                "weight_by_count":    ("BOOLEAN", {"default": False,                                  "tooltip": "use the count column as sampling weight — characters with more posts are more likely to be picked"}),
                "seed":               ("INT",     {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "random seed — same seed produces the same selection"}),
                "include_core_tags":  ("BOOLEAN", {"default": False,                                  "tooltip": "append each character's core descriptive tags (e.g. hair color, eye color) after their trigger"}),
                "include_copyright":  ("BOOLEAN", {"default": False,                                  "tooltip": "append each character's copyright/series name after their tags"}),
                "exclude":            ("STRING",  {"default": "", "multiline": False,                 "tooltip": "comma-separated character names to exclude from the pool before sampling (case-insensitive, underscores and spaces are equivalent)"}),
                "filter":             ("STRING",  {"default": "", "multiline": False,                 "tooltip": "only consider characters whose name contains this substring (case-insensitive); leave empty for no filtering"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tags",)
    FUNCTION = "pick_random_characters"
    CATEGORY = "utils"

    def pick_random_characters(self, file_path: str, count: int, delimiter: str, replace_underscore: bool, escape_parens: bool, trailing_comma: bool, weight_by_count: bool, seed: int, include_core_tags: bool, include_copyright: bool, exclude: str, filter: str) -> tuple[str]:
        excluded = _parse_exclude(exclude)
        needle = _normalize(filter)
        with open(os.path.expanduser(file_path), newline="", encoding="utf-8") as f:
            rows = [row for row in csv.DictReader(f)
                    if (not needle or needle in _normalize(row.get("character", "")))
                    and _normalize(row.get("character", "")) not in excluded]

        rng = random.Random(seed)
        selected = _sample(rng, rows, count, weight_by_count)

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
                "file_path":          ("STRING",  {"default": "", "multiline": False,                 "tooltip": "path to a CSV file with artist, trigger, count, and url columns"}),
                "count":              ("INT",     {"default": 1, "min": 1, "max": 1000, "step": 1,    "tooltip": "number of artists to pick"}),
                "delimiter":          ("STRING",  {"default": ", ", "multiline": False,               "tooltip": "string used to join picked artist triggers"}),
                "replace_underscore": ("BOOLEAN", {"default": True,                                   "tooltip": "replace underscores with spaces in output"}),
                "escape_parens":      ("BOOLEAN", {"default": True,                                   "tooltip": "escape ( and ) as \\( and \\) to prevent ComfyUI from interpreting them as attention weights"}),
                "trailing_comma":     ("BOOLEAN", {"default": False,                                  "tooltip": "append a trailing comma to the result"}),
                "weight_by_count":    ("BOOLEAN", {"default": False,                                  "tooltip": "use the count column as sampling weight — artists with more posts are more likely to be picked"}),
                "seed":               ("INT",     {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "random seed — same seed produces the same selection"}),
                "prefix":             ("STRING",  {"default": "", "multiline": False,                 "tooltip": "string prepended to each artist trigger, e.g. 'artist:' or '@'"}),
                "exclude":            ("STRING",  {"default": "", "multiline": False,                 "tooltip": "comma-separated artist names to exclude from the pool before sampling (case-insensitive, underscores and spaces are equivalent)"}),
                "filter":             ("STRING",  {"default": "", "multiline": False,                 "tooltip": "only consider artists whose name contains this substring (case-insensitive); leave empty for no filtering"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tags",)
    FUNCTION = "pick_random_artists"
    CATEGORY = "utils"

    def pick_random_artists(self, file_path: str, count: int, delimiter: str, replace_underscore: bool, escape_parens: bool, trailing_comma: bool, weight_by_count: bool, seed: int, prefix: str, exclude: str, filter: str) -> tuple[str]:
        excluded = _parse_exclude(exclude)
        needle = _normalize(filter)
        with open(os.path.expanduser(file_path), newline="", encoding="utf-8") as f:
            rows = [row for row in csv.DictReader(f)
                    if (not needle or needle in _normalize(row.get("artist", "")))
                    and _normalize(row.get("artist", "")) not in excluded]

        rng = random.Random(seed)
        selected = _sample(rng, rows, count, weight_by_count)

        triggers = [prefix + _process_tag(row.get("trigger", ""), replace_underscore, escape_parens) for row in selected if row.get("trigger", "").strip()]
        result = delimiter.join(triggers)
        if trailing_comma:
            result += ","
        return (result,)
