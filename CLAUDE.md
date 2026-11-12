# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ComfyUI-Image-Saver is a ComfyUI custom node plugin that saves images with generation metadata compatible with Civitai. It supports PNG, JPEG, WebP, and AVIF formats, storing model, LoRA, and embedding hashes for proper resource recognition.

## Development Commands

### Testing
```bash
cd saver && python -m pytest
cd tests && python -m pytest
```
See "Testing" below for what each suite covers.

## Key Features

- **Label vs Custom**: `custom` is embedded in the A1111 metadata string (and also usable via `%custom` in filenames); `label` (Image Saver / Image Saver Simple) is filename/path-only via `%label` and independent of metadata, meant for values shared across multiple Saver nodes in a workflow (e.g. one timestamp fed into several savers)

## Testing

Tests are split into two independent suites, each with its own `pytest.ini`, because `saver/saver.py` has no ComfyUI dependency while `nodes.py`/`utils.py` do:

- `saver/test_saver.py` — image format/metadata writing (`save_image`), no ComfyUI dependency. Run with `cd saver && python -m pytest`.
- `tests/test_paths.py` — output-path/filename containment (`resolve_within_output`, `ImageSaver.save_images`). `tests/conftest.py` stubs ComfyUI's runtime modules to import `nodes.py`/`utils.py` outside ComfyUI. Run with `cd tests && python -m pytest`.

## Important Notes

- WebP workflow embedding is experimental
