# Copyright 2025 The HuggingFace Team. All rights reserved.s
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
import re
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk, Features,Sequence,Value
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from open_r1.trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer, HumanOmniVLGRPOTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

import nltk
# nltk.download('punkt')
# nltk.download('punkt_tab')
from nltk.tokenize import PunktSentenceTokenizer
tokenizer = PunktSentenceTokenizer()

SYNONYM_MAP = {
    "anger": "anger", "angry": "anger", "rage": "anger",
    "disgust": "disgust", "disgusted": "disgust",
    "fear": "fear", "afraid": "fear", "scared": "fear", "frightened": "fear",
    "happy": "happiness", "happiness": "happiness", "joy": "happiness", "joyful": "happiness", "glad": "happiness", "pleased": "happiness",
    "neutral": "neutral", "none": "neutral", "no_emotion": "neutral", "no emotion": "neutral",
    "sad": "sadness", "sadness": "sadness", "sorrow": "sadness", "unhappy": "sadness", "depressed": "sadness",
    "surprise": "surprise", "surprised": "surprise", "startled": "surprise",
    "contempt": "contempt", "contemptuous": "contempt",
    "anxiety": "anxiety", "anxious": "anxiety", "nervous": "anxiety", "worried": "anxiety",
    "helpless": "helplessness", "helplessness": "helplessness",
    "disappointment": "disappointment", "disappointed": "disappointment",
}

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )

    json_file_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the JSONL file used as training video dataset."},
    )

    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )


def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    videos = kwargs.get("video", "")
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    pattern = re.compile(r"<think>.*?</think>\s*<answer>(.*?)</answer>", flags=re.DOTALL)
    for video, content, sol in zip(videos, contents, solution):
        reward = 0.0
        m = pattern.search(content)
        if not m:
            # 不符合指定 pattern，reward 保持 0
            rewards.append(reward)
            continue

        answer_norm = m.group(1).strip().lower()
        sol_norm = re.sub(r"<.*?>", "", str(sol)).strip().lower()

        if answer_norm in SYNONYM_MAP:
            answer_norm = SYNONYM_MAP[answer_norm]

        if sol_norm in SYNONYM_MAP:
            sol_norm = SYNONYM_MAP[sol_norm]

        if answer_norm == sol_norm:
            reward = 1.0
                
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Video: {video}\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


function_weights = {
    "information extraction": 1.0,
    "information analysis": 1.0,
    "conclusion": 2.0,
    "others": 0.0
}

modality_weights = {
    "audio": 1.0,
    "visual": 1.0,
    "contextual": 1.0,
    "audio-contextual": 2.0,
    "visual-contextual": 2.0,
    "audio-visual": 2.0,
    "audio-visual-contextual": 3.0
}

def FGCE_weighted(completions, solution, **kwargs):

    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []

    pattern = re.compile(r"<think>.*?</think>\s*<answer>(.*?)</answer>", flags=re.DOTALL)
    for content, sol in zip(completion_contents, solution):
        m = pattern.search(content)
        if not m:
            rewards.append(0) # MARK
            continue
        answer_content = m.group(1).strip()
        answer_norm = answer_content.lower()
        sol_norm = re.sub(r"<.*?>", "", str(sol)).strip().lower()
        if answer_norm in SYNONYM_MAP:
            answer_norm = SYNONYM_MAP[answer_norm]
        if sol_norm in SYNONYM_MAP:
            sol_norm = SYNONYM_MAP[sol_norm]
        if answer_norm != sol_norm:
            rewards.append(0)  # MARK
            continue

        think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
        think_content = think_match.group(1).strip() if think_match else ""
        
        if not think_content:
            rewards.append(0)  # MARK
            continue

        sentences = tokenizer.tokenize(think_content)
        batch_results = call_qwen3omni_model(sentences, "function_eval", answer_content, score_mode="fine_grained", gen_explanation=False)
        
        total_weights = 0
        curr_total_score = 0.0
        for item in batch_results:
            try:
                w = function_weights[item["function"]] * modality_weights[item["modality"]]
                curr_total_score += item["score"] * w
                total_weights += w
            except:
                continue
        curr_avg_score = curr_total_score / total_weights if total_weights > 0 else 0.0
        curr_avg_score  = (curr_avg_score - 1.0) / 4.0
        rewards.append(curr_avg_score)
    return rewards


def FGCE_unweighted(completions, **kwargs):
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content in completion_contents:
        answer_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
        answer_content = answer_match.group(1).strip() if answer_match else ""

        think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
        think_content = think_match.group(1).strip() if think_match else ""
        
        if not think_content:
            rewards.append(0)  # MARK
            continue

        sentences = tokenizer.tokenize(think_content)
        batch_results = call_qwen3omni_model(sentences, "function_eval", answer_content, score_mode="fine_grained", gen_explanation=False)
        
        total_weights = 0
        curr_total_score = 0.0
        for item in batch_results:
            try:
                curr_total_score += item["score"]
                total_weights += 1
            except:
                continue
        curr_avg_score = curr_total_score / total_weights if total_weights > 0 else 0.0
        curr_avg_score  = (curr_avg_score - 1.0) / 4.0
        rewards.append(curr_avg_score)
    return rewards

import requests, time      


def call_qwen3omni_model(explanation_sentence_batch, api_name, predicted_emotion="neutral", score_mode="fine_grained", gen_explanation=False):
    POD_IP = os.environ.get("POD_IP")
    # print(f"Calling Qwen3-Omni model at POD_IP: {POD_IP}")
    data = {
        "explanation_sentence_batch": explanation_sentence_batch,
        "predicted_emotion": predicted_emotion,
        "gen_explanation": gen_explanation,
        "score_mode": score_mode,
        "temperature": 0.0,
        "max_tokens": 128,
        "top_p": 1.0,
    }
    url = f"http://{POD_IP}:8000/{api_name}"

    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            resp = requests.post(url, json=data, timeout=30)
            if resp.status_code != 200:
                raise ValueError(f"HTTP {resp.status_code}: {resp.text[:100]}")

            result = resp.json()
            if isinstance(result, list) and len(result) > 0:
                return result
            else:
                raise ValueError("Empty or invalid JSON list returned")
        except Exception as e:
            wait = 2 ** attempt
            print(f"[call_batch_message] attempt {attempt+1} failed: {e}")
            time.sleep(wait)
    print("[call_batch_message] All retries failed, returning None")
    return None


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "FGCE_weighted": FGCE_weighted,
    "FGCE_unweighted": FGCE_unweighted,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

from datasets import Dataset, DatasetDict

def load_video_dataset(jsonl_path):
    # 读取jsonl文件
    with open(jsonl_path, 'r') as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines if line.rstrip()]
        data = [json.loads(line) for line in lines]

    # start_idx = int(600/1411*len(data))
    # data = data[start_idx:]
    # print(f"start_idx: {start_idx}")
    
    # 准备转换后的数据列表
    transformed_data = {
        'video': [],
        'problem': [],
        'solution': []
    }
    
    # 遍历json数据并转换
    for entry in data:
        video_path = entry['video']
        problem = None  # 初始化问题变量
        for conversation in entry['conversations']:
            if conversation['from'] == 'human':
              #  problem = conversation['value'].replace('<video>\n<audio>\n', '')
                problem = "As an emotional recognition expert; throughout the video, which emotion conveyed by the characters is the most obvious to you?"

            elif conversation['from'] == 'gpt' and problem is not None:
                solution = f"<answer> {conversation['value']} </answer>"
                # 添加到transformed_data
                transformed_data['video'].append(video_path)
                transformed_data['problem'].append(problem)
                transformed_data['solution'].append(solution)

    # 创建dataset
    dataset = Dataset.from_dict(transformed_data)
    dataset_dict = DatasetDict({'train': dataset})
    
    return dataset_dict


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load the dataset
    json_file_path = script_args.json_file_path
    if json_file_path is None:
        raise ValueError("You must provide --json_file_path path/to/file.jsonl")

    dataset = load_video_dataset(json_file_path)
   # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    
    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    # def make_conversation_image(example):
    #     return {
    #         "prompt": [
    #             {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
    #             {
    #                 "role": "user",
    #                 "content": [
    #                     {"type": "image"},
    #                     {"type": "text", "text": example["problem"]},
    #                 ],
    #             },
    #         ],
    #     }

    # QUESTION_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."

    QUESTION_TEMPLATE = "{Question} Output the thinking process in <think> </think> and final emotion in <answer> </answer> tags."
    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
        }
    
    def make_conversation_video(example):
        text = QUESTION_TEMPLATE.format(Question=example["problem"])
        # print(f"text: {text}")
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "video"},
                        {"type": "text", "text": text},
                    ],
                },
            ],
        }
    
    if "image" in dataset[script_args.dataset_train_split].features:
        print("has image in dataset")
        dataset = dataset.map(make_conversation_image)  # Utilize multiprocessing for faster mapping
        # dataset = dataset.remove_columns(["original_question", "original_answer"])
    elif "video" in dataset[script_args.dataset_train_split].features:
        print("has video in dataset")
        dataset = dataset.map(make_conversation_video)
    else:
        print("no image in dataset")
        dataset = dataset.map(make_conversation)
        dataset = dataset.remove_columns("messages")

    
   # trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainer
    trainer_cls = HumanOmniVLGRPOTrainer
    print("using: ", trainer_cls)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
