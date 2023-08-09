import torch
from safetensors.torch import load_file, save_file


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


device = torch.device("cpu")

diff_lora = torch.load("/xsl/wilson.xu/diffusers/sd-model-finetuned-lora/pytorch_lora_weights.bin", map_location=device)
diff_temp = {}
for k, v in diff_lora.items():
    if k.startswith('text_encoder'):
        temp = k.split('.')
        temp_key = '.'.join(temp[:5])
        if temp_key not in diff_temp:
            diff_temp[temp_key] = [k]
        else:
            diff_temp[temp_key].append(k)
    elif k.startswith('unet'):
        temp = k.split('.')
        if temp[1] == 'down_blocks' or temp[1] == 'up_blocks':
            temp_key = '.'.join(temp[:7])
        elif temp[1] == 'mid_block':
            temp_key = '.'.join(temp[:6])
        else:
            raise Exception(f'diff no {k}')

        if temp_key not in diff_temp:
            diff_temp[temp_key] = [k]
        else:
            diff_temp[temp_key].append(k)
    else:
        raise Exception(f'diff no {k}')

webui_lora = load_file("/xsl/wilson.xu/weights/handmix101.safetensors", device="cpu")
webui_temp = {}
for k, v in webui_lora.items():
    if k.startswith('lora_te'):
        temp = k.split('_')
        temp_key = '_'.join(temp[:7])
        if temp_key not in webui_temp:
            webui_temp[temp_key] = [k]
        else:
            webui_temp[temp_key].append(k)
    elif k.startswith('lora_unet'):
        temp = k.split('_')
        if temp[2] == 'down' or temp[2] == 'up':
            temp_key = '_'.join(temp[:7])
        elif temp[2] == 'mid':
            temp_key = '_'.join(temp[:6])
        else:
            raise Exception(f'webui no {k}')

        if temp_key not in webui_temp:
            webui_temp[temp_key] = [k]
        else:
            webui_temp[temp_key].append(k)
    else:
        raise Exception(f'webui no {k}')

w_lora = convert_lora(diff_lora)
save_file(w_lora, "./d2w_lora.safetensors")

print('Done!')
