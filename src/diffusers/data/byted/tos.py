import logging
import os
import io
import hashlib
import datetime
import uuid
import requests
import base64
import httpx
from PIL import Image
from retrying import retry
import bytedtos


bucket = "ad-creative-sg"
bucket_va = "ad-creative"
accessKey = "UZ4BEGULM9M9VCI165SG"
accessKey_va = "JRGOLSMK48IYLYWKM9DS"
logging.info(f"accessKey: {accessKey}")
url_prefix = "https://sf-tk-sg.ibytedtos.com/obj/" + bucket + "/"
url_prefix_va = "https://sf16-muse-va.ibytedtos.com/obj/" + bucket_va + "/"

tos_client = bytedtos.Client(bucket, accessKey)
tos_client_va = bytedtos.Client(bucket_va, accessKey_va)

VA_IDC_LIST = ["maliva", "Aliyun_VA", "USEAST3", "USEASTDT"]
IS_VA = os.getenv("merlin_debug", default="0") == "1" or os.getenv("TCE_INTERNAL_IDC") in VA_IDC_LIST


def generate_timestamp_uuid():
    # 获取当前时间的纳秒级时间戳
    nano_timestamp = datetime.datetime.fromtimestamp(datetime.datetime.now().timestamp()).timestamp()
    # 生成一个 UUID
    unique_id = uuid.uuid4()
    # 返回时间戳和 UUID 的组合
    return nano_timestamp, unique_id


def _gen_name(folder_name):
    nano_ts, uid = generate_timestamp_uuid()
    return (
        folder_name + "/" + "_".join(str(nano_ts).split(".")) + "_" + str(uid) if folder_name and folder_name != "" else str(nano_ts) + "_" + str(uid)
    )


def save_file_to_tos(file_path: str, object_name: str) -> str:
    with open(file_path, "rb") as fd:
        return save_tos(fd, object_name)


@retry(stop_max_attempt_number=2, wait_fixed=300)
def save_tos(content, object_name=None, folder_name="", overwrite=True, va=False, headers=None):
    # h["Content-Type"] = "image/jpeg"
    # h["Content-Type"]="text/html; charset=utf-8"
    if object_name is None:
        object_name = _gen_name(folder_name)
    logging.info(f"save_tos: {object_name}")
    cli = tos_client if not va and not IS_VA else tos_client_va
    prefix_url = url_prefix if not va and not IS_VA else url_prefix_va
    # avoid overwrite
    if not overwrite:
        cli.get_object(object_name)
        return prefix_url + object_name

    cli.put_object(object_name, content, headers=headers)
    url = prefix_url + object_name
    return url


def get_file_from_tos(url):
    if url.startswith(url_prefix_va):
        cli = tos_client_va
        prefix_url = url_prefix_va
    else:
        cli = tos_client
        prefix_url = url_prefix

    object_name = url.replace(prefix_url, "")
    try:
        result = cli.get_object(object_name)
    except Exception:
        raise Exception('Error: get object failed.')

    return result


def make_key(psm="ad.creative.image_core_solution"):
    """
    生成 tos 存储的 key，形如：201804085d0d982eb246e3b5cba746e8
    psm 可自选
    """
    psm_hash = hashlib.md5(psm.encode()).hexdigest()[0:4]
    key = uuid.uuid4().hex[0:20]
    date = datetime.datetime.now().strftime("%Y%m%d")
    return "{}{}{}".format(date, psm_hash, key)


def url_to_bytes(url: str) -> bytes:
    try:
        with requests.Session() as session:
            response = session.get(url, timeout=10)  # 可选：增加 timeout 避免长时间挂起
            response.raise_for_status()  # 抛出非 2xx 状态码的异常
            return response.content
    except requests.RequestException as e:
        logging.error(f"Request error occurred while requesting {url!r}: {e}")
        return None


def bytes_to_image(content_bytes: bytes) -> Image.Image:
    try:
        # 将字节转换为 BytesIO 对象
        image_data = io.BytesIO(content_bytes)
        # 使用 PIL.Image 打开图像
        image = Image.open(image_data)
        return image
    except IOError as e:
        logging.error(f"An error occurred while opening the image: {e}")
        return None


def url_to_base64(url: str, gpt_prefix: bool = False) -> str:
    """
    将 图片URL 转换为 base64 编码
    :param url: URL 地址
    :param gpt_prefix: 是否返回GPT标准base64格式
    :return: base64 编码
    """
    image_bytes = url_to_bytes(url)
    if image_bytes is None:
        return None

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    if gpt_prefix:
        image_pil = bytes_to_image(image_bytes)
        format = image_pil.format
        if format:
            format = format.lower()
        else:
            format = "png"
        return f"data:image/{format};base64,{image_base64}"
    else:
        return image_base64


def base64_to_md5(base64_str):
    # 计算 base64 字符串的 MD5 哈希值
    md5_hash = hashlib.md5(base64_str.encode("utf-8")).hexdigest()
    return md5_hash


def get_image_size_mb(image_url: str) -> float:
    size_mb = 0.0
    response = requests.head(image_url, timeout=10)
    if "content-length" in response.headers:
        size_mb = int(response.headers["content-length"]) / (1024 * 1024)

    return size_mb


# url_to_image_simple 不需要下载到本地的情况，直接从url获取
def url_to_image_simple(url: str) -> Image.Image:
    response = None
    image_data = None
    try:
        # 使用 httpx 发送请求
        with httpx.Client() as client:
            response = client.get(url, timeout=10)
            response.raise_for_status()

            # 关键优化：复制响应内容到新缓冲区，切断与response的关联
            image_data = io.BytesIO()
            image_data.write(response.content)  # 复制字节数据
            image_data.seek(0)  # 重置读写指针

            # 打开图像后立即复制，解除对image_data的依赖
            with Image.open(image_data) as temp_img:
                image = temp_img.copy()  # 复制图像数据到新对象
            return image  # 返回独立的图像对象，不依赖任何临时资源
    except httpx.RequestError as e:
        print(f"An error occurred while fetching the image from {e.request.url!r}: {e}")
        return None
    except httpx.HTTPStatusError as e:
        print(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
        return None
    except Exception as e:
        print(e)
        return None
    finally:
        # 显式清理response内部资源
        if response is not None:
            response._content = b""  # 清空响应体缓存
            if hasattr(response, "close"):
                response.close()
        # 清理临时变量
        del response
        del image_data


def image_to_bytes(image: Image.Image, convert_mode: bool = True) -> bytes:
    """
    将 PIL.Image 转换为字节
    :param image: PIL.Image 对象
    :param convert_mode: 是否需要转换模式
        convert_mode = True时，所有包含Alpha通道的图片会被转换为RGBA，不包含Alpha通道的图会被转化为RGB
    :return: 字节数据
    """

    try:
        # 创建 BytesIO 对象
        image_bytes = io.BytesIO()
        if convert_mode:
            if image.mode in ["RGB", "RGBA"]:
                pass
            elif image.mode in ["CMYK", "1", "L"]:
                image = image.convert("RGB")
            elif image.mode == "P":  # 调色版模式，也可以有Alpha通道
                if image.info.get("transparency", None):
                    image = image.convert("RGBA")
                else:
                    image = image.convert("RGB")
            else:
                image = image.convert("RGB")
        # 将图像保存到 BytesIO 对象
        image.save(image_bytes, format="PNG")
        # 获取字节数据
        image_bytes = image_bytes.getvalue()
        return image_bytes
    except IOError as e:
        logging.error(f"An error occurred while saving the image: {e}")
        return None


def resize_image(
    image_url: str,
    max_size: tuple = (480, 480),
    save_mode: int = 0,
    tos_dir: str = "carousel/compressed",
    local_dir: str = "resized_images",
    local_filename: str = "",
) -> str:
    image = url_to_image_simple(image_url)
    ori_size = image.size  # (width, height)
    if ori_size[0] > max_size[0] or ori_size[1] > max_size[1]:
        image.thumbnail(max_size)

    if save_mode == 0:  # save locally
        os.makedirs(local_dir, exist_ok=True)
        if not local_filename:
            local_filename = f"{uuid.uuid4()}.png"
        local_path = os.path.join(local_dir, local_filename)
        image.save(local_path)
        resized_image_path = local_path
    elif save_mode == 1:  # save to url
        image_bytes = image_to_bytes(image)
        resized_image_path = save_tos(image_bytes, make_key(), folder_name=tos_dir) or ""
    else:
        raise ValueError(f"[resize_image] save_mode {save_mode} is not supported")

    return resized_image_path


if __name__ == "__main__":
    url = "https://d3rekvgx2f3gtb.cloudfront.net/images/webp/redesign-v4/home-v4/legacy/testimonial-bg@2x.webp"
    print(f"tos_url: {save_tos(url_to_bytes(url), folder_name='v2c')}")
