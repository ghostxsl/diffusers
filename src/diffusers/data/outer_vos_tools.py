# Copyright (c) wilson.xu. All rights reserved.
import requests
import json
import base64
import hashlib
import time
import traceback
from PIL import Image
from io import BytesIO


__all__ = [
    'encode_pil_bytes', 'decode_pil_bytes',
    'download_pil_image', 'upload_pil_image',
]


requests.packages.urllib3.disable_warnings()

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2",
}


def encode_pil_bytes(img, to_string=True, format='JPEG', quality=90):
    buf = BytesIO()
    img.save(buf, format=format, quality=quality)
    img_bytes = buf.getvalue()
    if to_string:
        return base64.b64encode(img_bytes).decode('utf-8')
    return img_bytes


def decode_pil_bytes(img_bytes, from_string=True):
    if from_string:
        img_bytes = base64.b64decode(img_bytes)
    return Image.open(BytesIO(img_bytes))


def download_pil_image(url, retry_times=3):
    for i in range(retry_times):
        try:
            content = requests.get(url, timeout=3, headers=HEADERS).content
            image = Image.open(BytesIO(content))
            return image
        except Exception as e:
            if i == retry_times - 1:
                raise Exception(traceback.format_exc())
            else:
                continue


def _md5_sign(params, secret):
    myhash = hashlib.md5()
    sort_param = sorted(params.items(), key=lambda x: x[0], reverse=False)
    md5_str = ''.join([str(_[1]) for _ in sort_param])
    md5_str = md5_str + str(secret)
    myhash.update(md5_str.encode(encoding='utf-8'))
    return myhash.hexdigest()


def upload_pil_image(img_path, img_data, retry_times=3):
    IMAGE_API_URL = "http://image.api.vipshop.com/gw.php"
    api_key = 'virtualfitting'
    secret = '3a9239fdb4034fee'
    header = {'Content-Type': 'application/x-www-form-urlencoded;charset=utf-8'}

    img_string = encode_pil_bytes(img_data)
    for i in range(retry_times):
        try:
            time_str = time.strftime('%Y/%m%d/%H/%M/%S')
            param = {
                'ver': '1.0.0',
                'service': 'Process.uploadImg',
                'overwrite': 1,
                'sub_folder': 'vip-airc/{}'.format(time_str),  # test
                'api_key': api_key,
                'filename': img_path,
                'file_content': img_string,
            }
            # 生成api_sign的值
            md5_result = _md5_sign(param, secret)
            param['api_sign'] = md5_result
            res = requests.post(IMAGE_API_URL, headers=header, data=param, timeout=(10, 20))
            if res and res.text:
                json_data = json.loads(res.text)
                if json_data['code'] == 200 and 'result' in json_data and 'url' in json_data['result']:
                    return json_data['result']['url']
        except Exception as e:
            if i == retry_times - 1:
                raise Exception(traceback.format_exc())
            else:
                continue


if __name__ == "__main__":
    img_url = "https://a.vpimg2.com/upload/virtualfitting/vip-airc/2023/1227/21/41/59/ci.xiaohongshu.com_41915e5f-f955-5895-2364-0e576066ac4.jpg"

    im_data = download_pil_image(img_url)
    a = upload_pil_image("test.jpg", im_data)

    print(a)
    print("Done!")
