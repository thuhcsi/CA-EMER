import os
import copy
import warnings
import shutil
from functools import partial

import torch

from .model import load_pretrained_model
from .mm_utils import process_image, process_video, process_audio,tokenizer_multimodal_token, get_model_name_from_path, KeywordsStoppingCriteria,process_image_npary
from .constants import NUM_FRAMES, DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN, MODAL_INDEX_MAP, DEFAULT_AUDIO_TOKEN
import transformers

def model_init(model_path=None, **kwargs):
    # with_face = kwargs.get('with_face', False)
    model_path = "HumanOmni_7B" if model_path is None else model_path
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, processor, context_len, audio_processor = load_pretrained_model(model_path, None, model_name, **kwargs)

    if tokenizer.pad_token is None and tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token

    num_frames = model.config.num_frames if hasattr(model.config, "num_frames") else NUM_FRAMES
    if "qwen2vit" in model_path:
        from .mm_utils import process_image_qwen, process_video_qwen
        processor = {
            'image': partial(process_image_qwen, processor=processor, aspect_ratio=None),
            'video': partial(process_video_qwen, processor=processor, aspect_ratio=None, num_frames=num_frames),
        } 
    else:
        processor = {
                'image': partial(process_image, processor=processor, aspect_ratio=None),
                'video': partial(process_video, processor=processor, aspect_ratio=None, num_frames=num_frames),
                'face': partial(process_image_npary, processor=processor, aspect_ratio=None),
                'audio': partial(process_audio, processor=audio_processor),
            }
    return model, processor, tokenizer


def mm_infer(image_or_video, instruct, model, tokenizer, audio=None, modal='video', question=None, bert_tokeni=None, **kwargs):
    """inference api of HumanOmni for video understanding.

    Args:
        model: HumanOmni model.
        image_or_video (torch.Tensor): image tensor (1, C, H, W) / video tensor (T, C, H, W).
        instruct (str): text instruction for understanding video.
        tokenizer: tokenizer.
        do_sample (bool): whether to sample.
        modal (str): inference modality.
    Returns:
        str: response of the model.
    """
    question_prompt = None
    if question is not None:
        question = [question]
        question_prompt = bert_tokeni(question, return_tensors='pt', padding=True, truncation=True,add_special_tokens=True)
        question_prompt = {key: value.to('cuda') for key, value in question_prompt.items()}

    if modal == 'image':
        modal_token = DEFAULT_IMAGE_TOKEN
    elif modal == 'video':
        modal_token = DEFAULT_VIDEO_TOKEN
    elif modal == 'audio':
        modal_token = DEFAULT_AUDIO_TOKEN
    elif modal == 'video_audio':
        modal_token = DEFAULT_VIDEO_TOKEN + '\n' +DEFAULT_AUDIO_TOKEN
    elif modal == 'text':
        modal_token = ''
    else:
        raise ValueError(f"Unsupported modal: {modal}")


    # 1. vision preprocess (load & transform image or video).

    if modal == 'text' or modal == 'audio':
        tensor = [(torch.zeros(32, 3, 384, 384).cuda().half(), "video")]
    else:
        if "video" in modal:
            vi_modal = "video"
        else:
            vi_modal = "image"

        if isinstance(image_or_video, transformers.image_processing_base.BatchFeature):
            # 处理 BatchFeature 中的所有 tensor
            processed_data = transformers.image_processing_base.BatchFeature({
                'pixel_values_videos': image_or_video['pixel_values_videos'][0].half().cuda(),
                'video_grid_thw': image_or_video['video_grid_thw'][0].cuda()
            })
        else:
            # 处理普通 tensor
            processed_data = image_or_video.half().cuda()
        tensor = [(processed_data, vi_modal)]

    
    if audio is not None:
        audio = audio.half().cuda()

    # 2. text preprocess (tag process & generate prompt).
    if isinstance(instruct, str):
        message = [{'role': 'user', 'content': modal_token + '\n' + instruct}]
    elif isinstance(instruct, list):
        message = copy.deepcopy(instruct)
        message[0]['content'] = modal_token + '\n' + message[0]['content']
    else:
        raise ValueError(f"Unsupported type of instruct: {type(instruct)}")


    
    if model.config.model_type in ['HumanOmni', 'HumanOmni_mistral', 'HumanOmni_mixtral']:
        system_message = [
            {'role': 'system', 'content': (
            """<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."""
            """\n"""
            """If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>""")
            }
        ]
    else:
        system_message = []

    message = system_message + message
    prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

    # add modal warpper tokken
    if model.config.mm_use_x_start_end:
        prompt = prompt.replace("<video>", "<vi_start><video><vi_end>").replace("<image>", "<im_start><image><im_end>").replace("<audio>", "<au_start><audio><au_end>")
    prompt += kwargs.get("forced_think_text", "")

    input_ids = tokenizer_multimodal_token(prompt, tokenizer, modal_token, return_tensors='pt').unsqueeze(0).long().cuda()
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

    # 3. generate response according to visual signals and prompts. 
    keywords = [tokenizer.eos_token]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    do_sample = kwargs.get('do_sample', False)
    temperature = kwargs.get('temperature', 0.2 if do_sample else 0.0)
    top_p = kwargs.get('top_p', 1.0)
    max_new_tokens = kwargs.get('max_new_tokens', 2048)


    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_masks,
            images=tensor,
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.eos_token_id,
            prompts=question_prompt,
            audios=audio
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return outputs

def mm_infer_batch(image_or_video, instruct, model, tokenizer, audio=None, modal='video', question=None, bert_tokeni=None, **kwargs):
    """Batch inference for HumanOmni.
    """
    B = len(instruct)
    question_prompt = None
    if question is not None:
        question_prompt = bert_tokeni(question, return_tensors='pt', padding=True, truncation=True, add_special_tokens=True)
        question_prompt = {k: v.to('cuda') for k, v in question_prompt.items()}

    if modal == 'image':
        modal_token = DEFAULT_IMAGE_TOKEN
    elif modal == 'video':
        modal_token = DEFAULT_VIDEO_TOKEN
    elif modal == 'audio':
        modal_token = DEFAULT_AUDIO_TOKEN
    elif modal == 'video_audio':
        modal_token = DEFAULT_VIDEO_TOKEN + '\n' +DEFAULT_AUDIO_TOKEN
    elif modal == 'text':
        modal_token = ''
    else:
        raise ValueError(f"Unsupported modal: {modal}")


    # --- vision preprocess for batch ---
    images = None  # will become a list with length B or a BatchFeature-like object depending on model expectation
    vi_modal = "video" if "video" in modal else "image"

    if modal == 'text' or modal == 'audio':
        images = [(torch.zeros(32, 3, 384, 384).cuda().half(), "video")] * B
    else:
        # move to half and cuda
        img = image_or_video.half().cuda()
        images = []
        for i in range(B):
            images.append((img[i], vi_modal))

    
    # --- audio preprocess for batch ---
    audio = audio.half().cuda()
    if audio.dim() == 4 and audio.size(1) == 1:
        # remove the singleton channel dim -> shape [B, 128, 3000]
        audio = audio.squeeze(1)

    # --- text preprocess: construct messages/prompts for each batch item ---
    prompts = []
    for i in range(B):
        message = [{'role': 'user', 'content': modal_token + '\n' + instruct[i]}]
        # add system message if needed
        if model.config.model_type in ['HumanOmni', 'HumanOmni_mistral', 'HumanOmni_mixtral']:
            system_message = [
                {'role': 'system', 'content': (
                    "<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."
                    "\n"
                    "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>"
                )}
            ]
            full_message = system_message + message
        else:
            full_message = message
        prompts.append(tokenizer.apply_chat_template(full_message, tokenize=False, add_generation_prompt=True))

    # add forced_think_text per-item if provided (same text appended to all)
    forced_think_text = kwargs.get("forced_think_text", "")
    if forced_think_text:
        prompts = [p + forced_think_text for p in prompts]

    # if mm_use_x_start_end needed, apply to each prompt string
    if getattr(model.config, 'mm_use_x_start_end', False):
        prompts = [p.replace("<video>", "<vi_start><video><vi_end>").replace("<image>", "<im_start><image><im_end>").replace("<audio>", "<au_start><audio><au_end>") for p in prompts]

    # --- tokenize prompts into input_ids (per-item) and stack ---
    input_id_list = []
    for p in prompts:
        ids = tokenizer_multimodal_token(p, tokenizer, modal_token, return_tensors='pt')  # expect 1D tensor
        # ensure shape (seq_len,) -> unsqueeze batch dim
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        input_id_list.append(ids.long().cuda())

    # pad/stack to common sequence length (simple approach: pad to max len)
    max_len = max(x.shape[1] for x in input_id_list)
    padded_ids = []
    for ids in input_id_list:
        if ids.shape[1] < max_len:
            pad_len = max_len - ids.shape[1]
            pad_tensor = torch.full((1, pad_len), tokenizer.pad_token_id, dtype=torch.long).cuda()
            ids = torch.cat([ids, pad_tensor], dim=1)
        padded_ids.append(ids)
    input_ids = torch.cat(padded_ids, dim=0)  # (B, max_len)
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

    # --- stopping criteria ---
    keywords = [tokenizer.eos_token]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    # sampling params
    do_sample = kwargs.get('do_sample', False)
    temperature = kwargs.get('temperature', 0.2 if do_sample else 0.0)
    top_p = kwargs.get('top_p', 1.0)
    max_new_tokens = kwargs.get('max_new_tokens', 2048)

    # --- call model.generate in inference mode ---
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_masks,
            images=images,
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.eos_token_id,
            prompts=question_prompt,
            audios=audio
        )

    # decode: model.generate returns (B, seq_len) or list — use tokenizer.batch_decode
    if isinstance(output_ids, torch.Tensor):
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    else:
        # if model returns list of id lists
        outputs = [tokenizer.decode(o, skip_special_tokens=True) if isinstance(o, (list, torch.Tensor)) else str(o) for o in output_ids]

    # strip whitespace and return list
    outputs = [o.strip() for o in outputs]
    return outputs