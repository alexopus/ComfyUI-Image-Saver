# Examples

## [`example-workflow.json`](./example-workflow.json)

This is the standard example workflow demonstrating the basic usage of the image saver node to save images along with their generation metadata (including hashes for Civitai).

## [`image_saver_pipe_workflow.json`](./image_saver_pipe_workflow.json)

This workflow demonstrates how to leverage a "Pipe" architecture to efficiently construct and manage multiple image generation scenarios without the clutter of endless noodle-wiring.

**Key Features & Concepts:**
- **Image Saver Pipe Architecture:** Instead of manually routing dozens of individual connections (e.g., seed, steps, CFG, sampler name, paths, custom text) to every single save node, this workflow bundles all your metadata and generation settings into a unified `IMAGESAVER_PIPE` connection.
- **Impact Pack Integration:** This workflow is designed to work in tandem with the `basic_pipe` from the [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack), which similarly bundles your Model, CLIP, VAE, and Conditioning.
- **Workflow Expansion Tip:** While not required, this pipeline approach pairs beautifully with text-generation nodes like [comfyui-adaptiveprompts](https://github.com/Alectriciti/comfyui-adaptiveprompts). You can easily route a prompt "context" string alongside your `basic_pipe` and `IMAGESAVER_PIPE` to drive highly dynamic, multi-scenario generations.

**How it Works:**
1. **Initialization:** The base parameters are configured and bundled early in the workflow using the `Make Image Saver Pipe` node alongside Impact Pack's `ToBasicPipe`.
2. **Branching (Scenarios):** The unified pipe is then routed into different distinct scenarios or sub-graphs (e.g., variations in prompt, character, or setting).
3. **Editing on the Fly:** By using the `Edit Image Saver Pipe` node, you can selectively override specific metadata (like a scenario name or specific prompt additions) for a particular branch, all while inheriting the rest of the base settings.
4. **Inspection & Saving:** The `Read Image Saver Pipe` node allows you to extract and inspect the pipe's current values. Finally, the `Image Saver (From Pipe)` node takes the generated image and the pipe to save your output with all the accurately tracked, scenario-specific metadata attached.
