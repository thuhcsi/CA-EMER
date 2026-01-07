# coding: utf-8
import os
import json
import re
from typing import Dict, Any, Tuple, Optional, List
from pydantic import BaseModel
import uvicorn
import torch
from fastapi import FastAPI

from vllm import LLM, SamplingParams
from transformers import Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", "-p", type=str, required=True, help="Path to Qwen3-Omni-30B-A3B-Instruct judge model.")
args = parser.parse_args()

tp = torch.cuda.device_count() if torch.cuda.is_available() else 1
os.environ.setdefault('VLLM_USE_V1', '0')

model = LLM(
    model=args.model_path,
    trust_remote_code=True,
    gpu_memory_utilization=0.95,
    tensor_parallel_size=tp,
    limit_mm_per_prompt={'image': 3, 'video': 3, 'audio': 10},
    max_num_seqs=8,
    max_model_len=44000,
    seed=1234,
)

processor = Qwen3OmniMoeProcessor.from_pretrained(args.model_path, trust_remote_code=True)

app = FastAPI(title="Qwen3-Omni-VLLM Online Service with Audio Evaluation")


class InferenceRequest(BaseModel):
    explanation_sentence_batch: List[str] = []
    predicted_emotion: str = ""
    score_mode: str = None
    gen_explanation: bool = False
    full_explanation_content: str = ""
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 1
    max_tokens: int = 32
    use_audio_in_video: bool = False
    return_audio: bool = False


def build_input(processor, messages, use_audio_in_video: bool):
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio_in_video)
    
    inputs = {
        "prompt": text,
        "multi_modal_data": {},
        "mm_processor_kwargs": {"use_audio_in_video": use_audio_in_video},
    }
    if audios is not None:
        inputs['multi_modal_data']['audio'] = audios
    if images is not None:
        inputs['multi_modal_data']['image'] = images
    if videos is not None:
        inputs['multi_modal_data']['video'] = videos
    
    return inputs

def extract_json(raw_response_text: str):
    try:
        cleaned_response = raw_response_text.strip()
        
        if "```json" in cleaned_response:
            start = cleaned_response.find("```json") + len("```json")
            end = cleaned_response.find("```", start)
            if end == -1:
                end = len(cleaned_response)
            cleaned_response = cleaned_response[start:end].strip()
        elif cleaned_response.startswith("```") and cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[3:-3].strip()
            if cleaned_response.startswith("json"):
                cleaned_response = cleaned_response[4:].strip()
        return json.loads(cleaned_response)
    except:
        return None


FUNCTIONS = ["information extraction","information analysis","conclusion","others"]

MODALITIES = ["audio","visual","contextual","audio-contextual","visual-contextual","audio-visual","audio-visual-contextual"]

EVAL_PROMPT_FUNCTION = """
You are an evaluator. Read the Full Explanation Content and the Target Explanation Step, then return a STRICT JSON object only.

Functions: ["information extraction","information analysis","conclusion","others"]
Modalities: ["audio","visual","contextual","audio-contextual","visual-contextual","audio-visual","audio-visual-contextual"]

# Decision rule for modality (if sentence references multiple modalities):
1. If the sentence explicitly cites evidence from multiple modalities, choose the modality that provides the **most direct evidence** for the claim.
2. If both modalities are equally cited, choose the **more specific compound modality** (e.g., choose "audio-visual" over "audio").
3. If no modality is clearly cited, choose "contextual".

# Tips:
- "information extraction" means extracting cues without emotional words or emotion-centric analysis
- "information analysis" means the sentence contains emotional words or emotion-centric analysis
- "contextual" includes: ASR, OCR, speech content, subtitle, contextual explanation and etc.
- "audio" includes: pitch, speaking rate, volume, hesitation, tone, prosody, and etc.
- "visual" includes: facial expression, gaze, gesture, posture, and etc.

# Only output strict JSON:
{{
    "function": str(one of Functions),
    "modality": str(one of Modalities),
    "confidence": <float 0.0-1.0>
}}

# Inputs
Full Explanation Content:
{Full_Explanation_Content}

Target Explanation Step:
{Explanation}
"""

EVAL_PROMPT_SUB_FUNCTION = {
"information extraction": """
You are an impartial evaluator.
Your task is to evaluate the Information Extraction Sentence based on:
1. Whether it is a necessary or reasonable step in the processing pipeline.
2. How much emotional information it contributes for identifying the Predicted Emotion.

### Evaluation Criteria (Score: 1-5)

**1 - Unnecessary / Irrelevant Step**
- Not required, provides no emotional information.
**2 - Necessary but Emotionally Deviated**
- Operationally valid but the extracted cue does not support the Predicted Emotion (or is misleading).
**3 - Weak Emotional Support**
- Provides weak/indirect emotional hints.
**4 - Strong Emotional Support**
- Clear, relevant cue that helps infer Predicted Emotion.
**5 - Critical Emotional Evidence**
- Direct decisive cue (e.g., explicit cry, explicit "I'm dying", clear facial expression tied to emotion, etc).

### Conservative Rule
If the input simultaneously meets both criteria for a low score and a high score, **prefer the lower score**.

### Inputs
Predicted Emotion: {Emotion}

Full Reasoning Content:
{Full_Explanation_Content}

Information Extraction Sentence:
{Explanation}
""",
"information analysis": """
You are an impartial evaluator.
Your task is to evaluate whether the Information Analysis Sentence provides **valid, grounded, and emotionally relevant analysis** based on the Full Reasoning Content.

### Evaluation Criteria (Score: 1-5)

**1 - Hallucinated / Invalid Analysis**
- Contradicts evidence or invents motives without basis.
**2 - Grounded but Emotionally Irrelevant Analysis**
- Factually grounded but adds no help for emotion inference.
**3 - Weak Emotional Contribution**
- Provides limited insight or ambiguous interpretation.
**4 - Strong Emotional Contribution**
- Clear, grounded, adds substantive support to emotion inference.
**5 - Critical Emotional Analysis**
- Decisive, explains causal link between cues and emotion thoroughly.

### Conservative Rule
If the input simultaneously meets both criteria for a low score and a high score, **prefer the lower score**.

### Inputs
Predicted Emotion: {Emotion}

Full Reasoning Content:
{Full_Explanation_Content}

Information Analysis Sentence:
{Explanation}
""",
"conclusion": """
You are an impartial emotion-judging evaluator. 
Your task is to score how well the Conclusion Sentence of a explanation process reflects the Predicted Emotion.

### Evaluation Criteria (Score 1-5)

**1 - Completely Inconsistent**: conclusion contradicts Predicted Emotion
**2 - Weak Relevance**: barely related or vague
**3 - Partially Related**: some correct signals but ambiguous
**4 - Mostly Consistent**: reflects Predicted Emotion with minor wording issues
**5 - Fully Consistent and Accurate**: clear, precise, matches Predicted Emotion

P.S. Synonyms or close paraphrases are acceptable.

### Conservative Rule
If the input simultaneously meets both criteria for a low score and a high score, **prefer the lower score**.

### Inputs
Predicted Emotion: {Emotion}
Conclusion Step Reasoning: {Explanation}
"""
}

EXPLANATION_FORMAT = {
"binary": """
### Required Output Format (JSON only)
{{
    "score": <0 or 1>,
    "explanation": "<1-2 sentence brief explanation>"
}}
""",
"fine_grained": """
### Required Output Format (JSON only)
{{
    "score": <integer from 1 to 5>,
    "explanation": "<1-2 sentence brief explanation>"
}}
"""
}

NO_EXPLANATION_FORMAT = {
"binary": """
### Required Output Format (JSON only)
{{
    "score": <0 or 1>
}}
""",
"fine_grained": """
### Required Output Format (JSON only)
{{
    "score": <integer from 1 to 5>
}}
"""
}


@app.post("/function_eval")
async def function_eval(request: InferenceRequest):
    responses: List[Optional[Dict[str, Any]]] = [None] * len(request.explanation_sentence_batch)
    full_explanation_content = " ".join(request.explanation_sentence_batch)

    # helper: chunk a list
    def chunk_list(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    # Build first-stage prompts (batch)
    first_stage_inputs = []
    index_map = []
    for i, explanation_step in enumerate(request.explanation_sentence_batch):
        try:
            eval_prompt = EVAL_PROMPT_FUNCTION.format(
                Full_Explanation_Content=full_explanation_content,
                Explanation=explanation_step
            )
            eval_input = [{"role": "user", "content": [{"type": "text", "text": eval_prompt}]}]
            eval_inputs = build_input(processor, eval_input, False)
            first_stage_inputs.append(eval_inputs)
            index_map.append(i)
        except Exception as e:
            responses[i] = {"error": f"build_input error: {str(e)}"}

    if len(first_stage_inputs) == 0:
        return responses

    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
    )

    default_chunk = min(128, len(first_stage_inputs))

    stage1_results = {}  # original_index -> {"function": str, "modality": str, "confidence": float, "raw_text": str}

    try:
        for chunk_inputs, chunk_idx_slice in zip(chunk_list(first_stage_inputs, default_chunk),
                                                 chunk_list(index_map, default_chunk)):
            # model.generate expects list of inputs; vllm will process them in parallel
            chunk_outputs = model.generate(chunk_inputs, sampling_params=sampling_params)
            # Iterate outputs and map back to original indices
            for out, orig_idx in zip(chunk_outputs, chunk_idx_slice):
                try:
                    text = out.outputs[0].text
                    text = text.lower()
                    extracted_json = extract_json(text)

                    stage1_results[orig_idx] = {}

                    function = extracted_json.get("function", None)
                    if not function in FUNCTIONS:
                        stage1_results[orig_idx]["function"] = None
                    else:
                        stage1_results[orig_idx]["function"] = function

                    modality = extracted_json.get("modality", None)
                    if not modality in MODALITIES:
                        stage1_results[orig_idx]["modality"] = None
                    else:
                        stage1_results[orig_idx]["modality"] = modality

                    confidence = extracted_json.get("confidence", None)
                    try:
                        confidence = float(confidence)
                    except:
                        confidence = None
                    stage1_results[orig_idx]["confidence"] = confidence

                    stage1_results[orig_idx]["raw_text"] = text
                except Exception as e:
                    stage1_results[orig_idx] = {"function": None, "modality": None, "confidence": None, "raw_text": f"parse error: {str(e)}"}
    except Exception as e:
        # batch generation failed entirely
        return [{"error": f"first-stage batch generation error: {str(e)}"}]

    # Prepare second-stage batch: collect items that require detailed scoring
    stage2_items = []  # list of dicts: {"orig_idx": i, "eval_input": build_input(...), "function": function}
    for i, explanation_step in enumerate(request.explanation_sentence_batch):
        # If earlier we set an error in responses (from build_input), skip
        if responses[i] is not None:
            continue

        df = stage1_results.get(i)
        if df is None:
            responses[i] = {"error": "No function detection result."}
            continue

        function = df["function"]
        raw_text = df["raw_text"]
        if function is None:
            # first stage couldn't identify
            responses[i] = {"error": "No valid function detected", "judge_output": raw_text}
            continue

        # Fill trivial 'others' directly
        if function == "others":
            responses[i] = {
                "function": "others",
                "score": 0,
                "explanation": "Not information extraction, analysis or conclusion."
            }
            continue

        # For other functions, prepare second-stage prompt
        try:
            if function == "information extraction":
                prompt_template = EVAL_PROMPT_SUB_FUNCTION["information extraction"]
            elif function == "information analysis":
                prompt_template = EVAL_PROMPT_SUB_FUNCTION["information analysis"]
            elif function == "conclusion":
                prompt_template = EVAL_PROMPT_SUB_FUNCTION["conclusion"]
            else:
                # Unknown function - treat as error
                responses[i] = {"error": f"Unrecognized function: {function}"}
                continue

            if request.gen_explanation:
                prompt_template += EXPLANATION_FORMAT[request.score_mode]
            else:
                prompt_template += NO_EXPLANATION_FORMAT[request.score_mode]

            eval_prompt_2 = prompt_template.format(
                Emotion=request.predicted_emotion,
                Full_Explanation_Content=full_explanation_content,
                Explanation=explanation_step
            )
            eval_input_2 = [{"role": "user", "content": [{"type": "text", "text": eval_prompt_2}]}]
            eval_inputs_2 = build_input(processor, eval_input_2, False)

            stage2_items.append({
                "orig_idx": i,
                "eval_input": eval_inputs_2,
                "function": function
            })
        except Exception as e:
            responses[i] = {"error": f"second-stage build_input error: {str(e)}"}
            continue

    # If no second-stage items
    if len(stage2_items) == 0:
        for i in range(len(request.explanation_sentence_batch)):
            if responses[i] is None:
                df = stage1_results.get(i)
                if df:
                    responses[i] = {**df, "note": "No second-stage required"}
                else:
                    responses[i] = {"error": "No function detected and no second-stage needed."}
        return responses

    # Batch second-stage generation
    second_inputs = [item["eval_input"] for item in stage2_items]
    second_orig_indices = [item["orig_idx"] for item in stage2_items]

    try:
        for chunk_inputs, chunk_orig_idx in zip(chunk_list(second_inputs, default_chunk),
                                               chunk_list(second_orig_indices, default_chunk)):
            # note: chunk_inputs is a list of eval_input objects
            chunk_outputs = model.generate(chunk_inputs, sampling_params=sampling_params)
            for out, orig_idx in zip(chunk_outputs, chunk_orig_idx):
                try:
                    text = out.outputs[0].text
                    extracted_json = extract_json(text)
                    score = extracted_json.get("score", None)
                    try:
                        score = int(score)
                    except:
                        score = None
                    if extracted_json:
                        responses[orig_idx] = {
                            "function": stage1_results[orig_idx]["function"],
                            "modality": stage1_results[orig_idx]["modality"],
                            "confidence": stage1_results[orig_idx]["confidence"],
                            "score": score,
                            "explanation": extracted_json.get("explanation", None)
                        }
                    else:
                        # parse failed
                        responses[orig_idx] = {
                            "function": stage1_results[orig_idx]["function"],
                            "modality": stage1_results[orig_idx]["modality"],
                            "confidence": stage1_results[orig_idx]["confidence"],
                            "score": None,
                            "explanation": None,
                            "raw_output": text
                        }
                except Exception as e:
                    responses[orig_idx] = {"error": f"second-stage postprocess error: {str(e)}"}
    except Exception as e:
        return [{"error": f"second-stage batch generation error: {str(e)}"}]

    # Final fallback: any response slot still None
    for i in range(len(responses)):
        if responses[i] is None:
            responses[i] = {"error": "unknown processing error for this index"}

    return responses


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
