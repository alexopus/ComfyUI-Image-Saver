"""
Edit Metadata: changes only what is edited in the A1111 parameters and keeps the rest as written.
"""
import pytest

from image_saver_under_test.nodes import Metadata
from image_saver_under_test.nodes_metadata_edit import EditMetadata
from image_saver_under_test.nodes_pipe import EditImageSaverPipe, apply_string_edit

RESOURCES = '[{"modelName":"Pony","modelVersionName":"V6","modelVersionId":290640,"type":"checkpoint"}]'
PARAMETERS = ("a cat, sitting\nNegative prompt: dog\n"
              'Steps: 30, Sampler: DPM++ 2M Karras, CFG scale: 6.5, Seed: 42, Size: 832x1216, Clip skip: 2, '
              'Model hash: ABCDEF1234, Model: pony, Hashes: {"LORA:style":"0123456789"}, Version: ComfyUI, '
              f'Civitai resources: {RESOURCES}')


def metadata(**overrides):
    fields = dict(modelname="pony", positive="a cat, sitting", negative="dog", width=832, height=1216, seed=42,
                  steps=30, cfg=6.5, sampler_name="dpmpp_2m", scheduler_name="karras", denoise=1.0, clip_skip=2,
                  custom="", additional_hashes="", ckpt_path="", a111_params=PARAMETERS,
                  final_hashes="pony:ABCDEF1234,style:0123456789")
    return Metadata(**(fields | overrides))


def edit(source=None, **kwargs):
    # Unconnected number inputs arrive as missing kwargs; text widgets default to "[original]"
    text_defaults = {"modelname": "[original]", "positive": "[original]", "negative": "[original]"}
    return EditMetadata().edit(source or metadata(), **(text_defaults | kwargs))[0]


def test_untouched_inputs_keep_everything():
    assert edit() == metadata()


def test_append_to_positive_keeps_settings_verbatim():
    result = edit(positive="[original], masterpiece")
    assert result.positive == "a cat, sitting, masterpiece"
    assert result.a111_params == PARAMETERS.replace("a cat, sitting\n", "a cat, sitting, masterpiece\n", 1)


def test_empty_negative_removes_the_line():
    result = edit(negative="")
    assert result.negative == ""
    assert "Negative prompt" not in result.a111_params
    assert result.a111_params.startswith("a cat, sitting\nSteps: 30,")


def test_connected_numbers_change_settings_and_fields():
    result = edit(seed=7, steps=12, cfg=4, width=1664, denoise=0.35, clip_skip=0)
    assert (result.seed, result.steps, result.cfg, result.width, result.height, result.denoise) == (7, 12, 4.0, 1664, 1216, 0.35)
    assert "Steps: 12, Sampler: DPM++ 2M Karras, CFG scale: 4.0, Seed: 7, Size: 1664x1216, Model hash:" in result.a111_params
    assert "Clip skip" not in result.a111_params
    # New keys go before "Version"
    assert ', Denoising strength: 0.35, Version: ComfyUI,' in result.a111_params
    assert result.a111_params.endswith(f"Civitai resources: {RESOURCES}")


def test_sampler_and_scheduler_use_civitai_names():
    result = edit(scheduler_name="exponential")
    assert (result.sampler_name, result.scheduler_name) == ("dpmpp_2m", "exponential")
    assert "Sampler: DPM++ 2M Exponential," in result.a111_params


def test_remove_settings_and_hashes():
    result = edit(remove_settings="civitai resources, Hashes, Model hash")
    assert result.a111_params.endswith("Clip skip: 2, Model: pony, Version: ComfyUI")
    assert result.final_hashes == ""
    assert edit(remove_settings="Version").final_hashes == metadata().final_hashes


def test_removing_hashes_keeps_model_hash():
    # "Model hash" is its own setting, so the checkpoint stays identified and stays in the hashes output
    result = edit(remove_settings="Hashes")
    assert "Model hash: ABCDEF1234, Model: pony, Version: ComfyUI" in result.a111_params
    assert result.final_hashes == "pony:ABCDEF1234"


def test_hashes_not_in_parameters_are_kept():
    # e.g. resources resolved via Civitai: in the hashes output, but not in the text
    result = edit(metadata(final_hashes="pony:ABCDEF1234,style:0123456789,other:FEDCBA9876:0.5"), remove_settings="Hashes")
    assert result.final_hashes == "pony:ABCDEF1234,other:FEDCBA9876:0.5"


def test_model_name_changes_only_the_model_entry():
    result = edit(modelname="sdxl/juggernaut.safetensors")
    assert result.modelname == "sdxl/juggernaut.safetensors"
    assert "Model hash: ABCDEF1234, Model: juggernaut, Hashes:" in result.a111_params


def test_metadata_without_parameters_only_changes_fields():
    result = edit(metadata(a111_params=""), seed=5, positive="new")
    assert (result.a111_params, result.seed, result.positive) == ("", 5, "new")


def test_no_metadata_passes_through():
    assert EditMetadata().edit(None) == (None,)


def test_inputs_match_edit_pipe():
    inputs = EditMetadata.INPUT_TYPES()
    pipe_inputs = EditImageSaverPipe.INPUT_TYPES()["optional"]
    assert inputs["required"] == {"metadata": ("METADATA", {"tooltip": "metadata to edit"})}
    assert inputs["optional"]["seed"] == pipe_inputs["seed"]
    assert inputs["optional"]["positive"][1]["default"] == "[original]"
    assert inputs["optional"]["sampler_name"][1].get("forceInput") is True


@pytest.mark.parametrize("new,expected", [(None, "x"), ("[original]", "x"), ("[original] y", "x y"), ("z", "z"), ("", "")])
def test_apply_string_edit(new, expected):
    assert apply_string_edit(new, "x") == expected


def test_remove_embeddings_keeps_model():
    # The case from the UI: new negative prompt without embeddings, and their hashes removed by name
    params = ("a bottle\nNegative prompt: embedding:ng_deepnegative_v1_75t, embedding:EasyNegative\n"
              "Steps: 20, Sampler: Euler, Seed: 1, Size: 512x512, Model hash: 4ecddc97d4, Model: helloart, "
              'Hashes: {"embed:ng_deepnegative_v1_75t":"54e7e4826d","embed:EasyNegative":"66a7279a88","model":"4ecddc97d4"}, '
              "Version: ComfyUI")
    source = metadata(a111_params=params, negative="embedding:ng_deepnegative_v1_75t, embedding:EasyNegative",
                      final_hashes="helloart:4ecddc97d4,ng_deepnegative_v1_75t:54e7e4826d,EasyNegative:66a7279a88")
    result = edit(source, negative="TESTMINUS", remove_resources="ng_deepnegative_v1_75t, easynegative")
    assert result.a111_params == ("a bottle\nNegative prompt: TESTMINUS\n"
                                  "Steps: 20, Sampler: Euler, Seed: 1, Size: 512x512, Model hash: 4ecddc97d4, Model: helloart, "
                                  'Hashes: {"model":"4ecddc97d4"}, Version: ComfyUI')
    assert result.final_hashes == "helloart:4ecddc97d4"


def test_remove_resources_without_parameters_still_filters_hashes_output():
    result = edit(metadata(a111_params=""), remove_resources="style")
    assert result.final_hashes == "pony:ABCDEF1234"
