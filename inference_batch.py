import os
import argparse
import json
import re
from pathlib import Path
from tqdm import tqdm

import torch
import torch.distributed as dist
import sys
sys.path.append("./src")
from humanomni import model_init, mm_infer_batch
from humanomni.utils import disable_torch_init
from transformers import BertTokenizer

class JsonlDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_path):
        self.jsonl_path = jsonl_path
        self._offsets = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            offset = 0
            for line in f:
                self._offsets.append(offset)
                offset += len(line.encode('utf-8'))
        self.size = len(self._offsets)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        offset = self._offsets[idx]
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            f.seek(offset)
            line = f.readline()
        item = json.loads(line)
        return item

class InferenceSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = dist.get_rank()
        self._world_size = dist.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def extract_tag(text, tag):
    if text is None:
        return None
    pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.IGNORECASE | re.DOTALL)
    m = pattern.search(text)
    if not m:
        return None
    return m.group(1).strip()

def parse_model_response(raw_resp, early):
    if raw_resp is None:
        return None, None
    if early:
        return None, raw_resp.replace("</answer>", "")
    think = extract_tag(raw_resp, "think")
    pred = extract_tag(raw_resp, "answer")
    return think, pred

def recursive_to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, (list, tuple)):
        return type(data)(recursive_to_device(item, device) for item in data)
    elif isinstance(data, dict):
        return {k: recursive_to_device(v, device) for k, v in data.items()}
    else:
        return data

def remove_accelerate_hooks(model):
    from accelerate.hooks import remove_hook_from_module

    remove_hook_from_module(model, recurse=True)
    
    def _remove_hooks(module):
        if hasattr(module, '_hf_hook'):
            delattr(module, '_hf_hook')
        if hasattr(module, '_old_forward'):
            if hasattr(module, '_accelerate_original_forward'):
                module.forward = module._accelerate_original_forward
                delattr(module, '_accelerate_original_forward')
    
    model.apply(_remove_hooks)
    return model

def fully_move_model_to_device(model, device):
    model = remove_accelerate_hooks(model)
    model.to(device)
    
    for name, param in model.named_parameters():
        if param.device != device:
            param.data = param.data.to(device)
    
    for name, buffer in model.named_buffers():
        if buffer.device != device:
            buffer.data = buffer.data.to(device)
    
    for module in model.modules():
        if hasattr(module, '_parameters'):
            for key, param in module._parameters.items():
                if param is not None and param.device != device:
                    module._parameters[key] = param.to(device)
        if hasattr(module, '_buffers'):
            for key, buffer in module._buffers.items():
                if buffer is not None and buffer.device != device:
                    module._buffers[key] = buffer.to(device)
    
    return model

def parse_args():
    parser = argparse.ArgumentParser(description="HumanOmni Distributed Inference")
    parser.add_argument('--modal', type=str, default='video_audio')
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--input_jsonl', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--bert_path', type=str, required=True)
    parser.add_argument('--early', action='store_true')
    return parser.parse_args()

def init_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        rank = 0
        world_size = 1
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
    
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')
    
    if world_size > 1:
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(backend=backend, init_method='env://')
    
    return rank, world_size, local_rank, device

def main():
    args = parse_args()

    if "7B" in args.model_path:
        args.batch_size = 8
    else:
        args.batch_size = 32
    
    rank, world_size, local_rank, device = init_distributed()
    
    print(f"Rank {rank}, Local Rank {local_rank}, Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    basename = os.path.basename(args.input_jsonl)
    if args.early:
        basename = basename.replace(".jsonl", "_early.jsonl")
    out_path = Path(args.output_dir) / basename
    print(f"out_path: {out_path}")
    
    if args.bert_path:
        bert_tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    else:
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    disable_torch_init()
    
    print(f"Rank {rank}: Loading model...")
    model, processor, tokenizer = model_init(args.model_path)
    
    print(f"Rank {rank}: Moving model to device {device}...")
    model = fully_move_model_to_device(model, device)
    model.eval()
    
    print(f"Rank {rank}: Verifying model device...")
    for name, param in model.named_parameters():
        if param.device != device:
            print(f"Warning: Parameter {name} is on {param.device}, not {device}")
    
    dataset = JsonlDataset(args.input_jsonl)
    
    if world_size > 1:
        sampler = InferenceSampler(len(dataset))
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)
    
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    if args.early:
        forced_think_text = "<think></think>\n<answer>"
    else:
        forced_think_text = ""
    
    print(f"Rank {rank}: Starting inference...")
    with open(out_path, 'w', encoding='utf-8') as fout:
        for batch_idx, item_batch in enumerate(tqdm(dataloader, desc=f'Rank{rank}', disable=(rank!=0))):
            instructions = []
            gts = []
            video_tensors = []
            audio_tensors = []
            videos = item_batch.get('video')
            conversations = item_batch.get('conversations')
            for i, video_path in enumerate(videos):
                instruction = conversations[0]["value"][i]
                gt = conversations[1]["value"][i]

                video_tensor = processor['video'](video_path)
                video_tensor = recursive_to_device(video_tensor, device)
                
                audio_tensor = processor['audio'](video_path)[0]
                audio_tensor = recursive_to_device(audio_tensor, device)

                instructions.append(instruction)
                gts.append(gt)
                video_tensors.append(video_tensor)
                audio_tensors.append(audio_tensor)

            video_tensors = torch.stack(video_tensors, dim=0)
            audio_tensors = torch.stack(audio_tensors, dim=0)
            
            try:
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=(torch.cuda.is_available())):
                    raw_outputs = mm_infer_batch(
                        video_tensors,
                        instructions,
                        model=model,
                        tokenizer=tokenizer,
                        modal=args.modal,
                        question=instructions,
                        bert_tokeni=bert_tokenizer,
                        do_sample=False,
                        audio=audio_tensors,
                        forced_think_text=forced_think_text
                    )
                
                for i, raw_output in enumerate(raw_outputs):
                    think_text, pred = parse_model_response(raw_output, early=args.early)
                    print(f"think_text: {think_text}\npred: {pred}\n")
                    rec = {
                        "video": videos[i],
                        "gt": gts[i],
                        "pred": pred,
                        "think_text": think_text,
                        "raw_response": raw_output,
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                
            except Exception as e:
                print(f"Rank {rank}, Error processing {video_path}: {str(e)}")
    
    print(f"Rank {rank}: Finished inference")
    
    if world_size > 1:
        dist.barrier()
        if rank == 0:
            print(f"All ranks finished. Outputs saved to {args.output_dir}")
        dist.destroy_process_group()
    
    if rank == 0:
        print(f"\nTo merge outputs:")
        print(f"cat {args.output_dir}/{os.path.basename(args.input_jsonl)}.rank* > {args.output_dir}/{os.path.basename(args.input_jsonl)}")

if __name__ == '__main__':
    main()