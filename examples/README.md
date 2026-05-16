# Examples

This directory contains example workflows to help you get started with the ComfyUI-Image-Saver nodes.

## [`example-workflow.json`](./example-workflow.json)

This is the **standard example workflow** and the recommended starting point for most users. 

It demonstrates a basic setup using the `Image Saver Metadata` and `Image Saver Simple` nodes. In this example:
- The image is saved in **JPEG** format with A1111-compatible metadata.
- It shows how to use **filename placeholders** like `%time` and `%seed`.
- It illustrates how **resource hashes** (like embeddings in the prompt) are automatically handled by the metadata engine.
- It includes helper nodes like `Input Parameters` and `String Literal` for better workflow organization.

## [`image_saver_pipe_workflow.json`](./image_saver_pipe_workflow.json)

This is an **optional advanced workflow** demonstrating how to use the Pipe system for orchestration. 

It is designed for complex scenarios where managing multiple generation branches would otherwise result in a cluttered "noodle" of wires. By bundling metadata and saver settings into a single `IMAGESAVER_PIPE` connection, you can:
- Reduce wire clutter in large-scale workflows.
- Create base configurations early and selectively override them in different branches using the `Edit Image Saver Pipe` node.
- Deferred execution of expensive metadata operations until the final save node.

This approach is particularly useful when working in tandem with other piping systems, such as the `Basic Pipe` from the [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack).
