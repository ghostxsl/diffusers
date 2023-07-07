import logging
import bytedtos
import os
import hashlib
import datetime
import uuid
import requests


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


def url_to_bytes(url: str) -> bytes:
    try:
        with requests.Session() as session:
            response = session.get(url, timeout=10)  # 可选：增加 timeout 避免长时间挂起
            response.raise_for_status()  # 抛出非 2xx 状态码的异常
            return response.content
    except requests.RequestException as e:
        logging.error(f"Request error occurred while requesting {url!r}: {e}")
        return None


def save_file_to_tos(file_path: str, object_name: str) -> str:
    with open(file_path, "rb") as fd:
        return save_tos(fd, object_name)


def _gen_name(folder_name):
    nano_ts, uid = generate_timestamp_uuid()
    return (
        folder_name + "/" + "_".join(str(nano_ts).split(".")) + "_" + str(uid) if folder_name and folder_name != "" else str(nano_ts) + "_" + str(uid)
    )


def save_tos(content, object_name=None, folder_name="", overwrite=True, va=False):
    if object_name is None:
        object_name = _gen_name(folder_name)
    logging.info(f"save_tos: {object_name}")
    cli = tos_client if not va and not IS_VA else tos_client_va
    prefix_url = url_prefix if not va and not IS_VA else url_prefix_va
    # avoid overwrite
    if not overwrite:
        try:
            cli.get_object(object_name)
            return prefix_url + object_name
        except Exception:
            pass

    try:
        cli.put_object(object_name, content)
        url = prefix_url + object_name
    except Exception as e:
        logging.error("upload obj to tos error: " + str(e))
        return None
    return url


def make_key(psm="ad.creative.image_core_solution"):
    """
    生成 tos 存储的 key，形如：201804085d0d982eb246e3b5cba746e8
    psm 可自选
    """
    psm_hash = hashlib.md5(psm.encode()).hexdigest()[0:4]
    key = uuid.uuid4().hex[0:20]
    date = datetime.datetime.now().strftime("%Y%m%d")
    return "{}{}{}".format(date, psm_hash, key)


if __name__ == "__main__":
    url = "https://d3rekvgx2f3gtb.cloudfront.net/images/webp/redesign-v4/home-v4/legacy/testimonial-bg@2x.webp"
    print(f"tos_url: {save_tos(url_to_bytes(url), folder_name='v2c')}")
