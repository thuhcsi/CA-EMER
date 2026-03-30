# Evaluating and Improving Explanation Coherence for Multimodal Emotion Recognition

[![Hugging Face](https://img.shields.io/badge/🤗HuggingFace-our_model-yellow)](https://huggingface.co/Anonymous23701/CA-EMER-HumanOmni-7B) [![Demo Page](https://img.shields.io/badge/👀-Demo_Page-blue)](https://thuhcsi.github.io/CA-EMER)

The official repository for "Evaluating and Improving Explanation Coherence for Multimodal Emotion Recognition".

Change all places with `YOUR_PATH` to your local directories.

## Preparations

### Environment
See `configs/HumanOmni.yml` for SFT (Python 3.10) and `configs/r1-v.yml` for GRPO (Python 3.11).

For the `qwen3_caption_vllm` environment, please refer to the [Qwen3-Omni repository](https://github.com/QwenLM/Qwen3-Omni).

### Pretrained Models

<div align="center">

| **Model** | **HuggingFace** |
|------------------------|---------------------------------------------------------------------------------|
|`HumanOmni-0.5B`|[![HF](https://img.shields.io/badge/🤗-Download-yellow)](https://hf.co/StarJiaxing/HumanOmni-0.5B)|
|`HumanOmni-7B`|[![HF](https://img.shields.io/badge/🤗-Download-yellow)](https://hf.co/StarJiaxing/HumanOmni-7B)|
|`Qwen3-Omni-30B-A3B-Instruct`|[![HF](https://img.shields.io/badge/🤗-Download-yellow)](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct)|
|`bert-base-uncased`|[![HF](https://img.shields.io/badge/🤗-Download-yellow)](https://huggingface.co/google-bert/bert-base-uncased)|
|`siglip-base-patch16-224`|[![HF](https://img.shields.io/badge/🤗-Download-yellow)](https://huggingface.co/google/siglip-base-patch16-224)|
|`siglip-so400m-patch14-384`|[![HF](https://img.shields.io/badge/🤗-Download-yellow)](https://huggingface.co/google/siglip-so400m-patch14-384)|
|`whisper-large-v3`|[![HF](https://img.shields.io/badge/🤗-Download-yellow)](https://huggingface.co/openai/whisper-large-v3)|
</div>


### Data Format

#### SFT
```json
{
    "video": "VIDEO_PATH",
    "conversations": [
        {
        "from": "human",
        "value": "<video>\n<audio>\nAs an emotional recognition expert; throughout the video, which emotion conveyed by the characters is the most obvious to you? Output the thinking process in <think> </think> and final emotion in <answer> </answer> tags."
        },
        {
        "from": "gpt",
        "value": "<think>THINK_CONTENT</think>\n<answer>EMOTION_LABEL</answer>"
        }
    ]
}
```

#### GRPO / Inference
```json
{
    "video": "VIDEO_PATH",
    "conversations": [
        {
            "from": "human",
            "value": "<video>\n<audio>\nAs an emotional recognition expert; throughout the video, which emotion conveyed by the characters is the most obvious to you? Output the thinking process in <think> </think> and final emotion in <answer> </answer> tags."
        },
        {
            "from": "gpt",
            "value": "EMOTION_LABEL"
        }
    ]
}
```

## Training

### SFT
```bash
bash srun_sft_humanomni.sh
```

### GRPO

To use FG-CE, run `srun_fgce.sh` and fill in the `POD_IP` in `srun_grpo_humanomni.sh`.

```bash
bash srun_grpo_humanomni.sh
```

## Inference

```bash
conda activate r1-v
torchrun --nproc_per_node=$GPUS --nnodes=1 \
    --master_addr=localhost --master_port=12345 \
    inference_batch.py \
    --model_path $MODEL_PATH \
    --bert_path $BERT_PATH \
    --input_jsonl $INPUT_JSONL \
    --output_dir $OUTPUT_DIR
```
