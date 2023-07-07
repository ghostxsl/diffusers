# Copyright (c) wilson.xu. All rights reserved.
import argparse
import torch
import pickle
from safetensors.torch import load_file, save_file


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a converting script.")
    parser.add_argument(
        "--convert_type",
        default='lora',
        type=str,
        choices=['lora', 'controlnet'],
        help="Convert model selection.")
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        type=str,
        required=True,
        help="Path to the checkpoint to convert.")
    parser.add_argument(
        "--from_torch",
        action="store_true",
        help="If `--checkpoint_path` is in `torch` format, load checkpoint with PyTorch.",
    )
    parser.add_argument(
        "--keys_map_path",
        default="controlnet_d2w_keys.map",
        type=str,
        help="Path to the mapping of keys to convert.")
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to the output model.")
    parser.add_argument("--device", default='cpu', type=str, help="Device to use (e.g. cpu, cuda:0, cuda:1, etc.)")

    args = parser.parse_args()
    return args


def pkl_load(file):
    with open(file, 'rb') as f:
        out = pickle.load(f)
    return out


def convert_lora(checkpoint, mode='d2w', lora_rank=32):
    out = {}
    if mode == 'd2w':
        webui_lora_alpha = torch.tensor(lora_rank, dtype=torch.float32)
        lora_merge = {}
        for k, v in checkpoint.items():
            key = k.split('.')
            if '.'.join(key[:-2]) not in lora_merge:
                lora_merge['.'.join(key[:-2])] = {k: v}
            else:
                lora_merge['.'.join(key[:-2])][k] = v
        # convert diffusers to webui
        for k, v in lora_merge.items():
            if k.startswith('text_encoder'):
                key = k.split('.')
                key[0] = 'lora_te'
                webui_key = '_'.join(key[:7])
                out[webui_key + '.alpha'] = webui_lora_alpha.clone()
                for k2, v2 in v.items():
                    if k2.split('.')[-2] == 'down':
                        out[webui_key + '.lora_down.weight'] = v2
                    elif k2.split('.')[-2] == 'up':
                        out[webui_key + '.lora_up.weight'] = v2
                    else:
                        raise Exception(f'error {k2} in text_encoder')
            elif k.startswith('unet'):
                key = k.split('.')
                key[0] = 'lora_unet'
                if key[1] == 'down_blocks' or key[1] == 'up_blocks':
                    webui_key = '_'.join(key[:8])
                    webui_key += '_'
                    webui_key += key[9].replace('_lora', '')
                elif key[1] == 'mid_block':
                    webui_key = '_'.join(key[:7])
                    webui_key += '_'
                    webui_key += key[8].replace('_lora', '')
                else:
                    raise Exception(f'error {k} in unet')
                if 'to_out' in webui_key:
                    webui_key += '_0'
                out[webui_key + '.alpha'] = webui_lora_alpha.clone()
                for k2, v2 in v.items():
                    if k2.split('.')[-2] == 'down':
                        out[webui_key + '.lora_down.weight'] = v2
                    elif k2.split('.')[-2] == 'up':
                        out[webui_key + '.lora_up.weight'] = v2
                    else:
                        raise Exception(f'error {k2} in unet')
    elif mode == 'w2d':
        pass
    else:
        raise Exception('')

    return out


def convert_controlnet(checkpoint, map_path, prefix='control_model.'):
    out = {}
    d2w_keys = pkl_load(map_path)
    for k, v in checkpoint.items():
        webui_key = d2w_keys.get(k, None)
        if webui_key is not None:
            webui_key = prefix + webui_key
            out[webui_key] = v
        else:
            raise Exception(f"No key: {k} in keys map")

    return out


def main(args):
    device = torch.device(args.device)

    if args.from_torch:
        diffusers_checkpoint = torch.load(args.checkpoint_path, map_location=device)
    else:
        diffusers_checkpoint = load_file(args.checkpoint_path, device=args.device)

    if args.convert_type == "lora":
        out_safetensors = convert_lora(diffusers_checkpoint, lora_rank=args.lora_rank)
        save_file(out_safetensors, args.dump_path)
    elif args.convert_type == "controlnet":
        out_pth = convert_controlnet(diffusers_checkpoint, args.keys_map_path)
        torch.save(out_pth, args.dump_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
    print('Done!')
