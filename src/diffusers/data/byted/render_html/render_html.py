import json
import time
from typing import Tuple

from diffusers.data.render_html.url2carousel_template import template as url2carouselTemplate
from diffusers.data.tos import save_tos


ErrCodeTosError = 51301
ErrorCodeRenderError = 52501


def renderUrl2CarouselSingleHtml(tosFolder: str, htmlFileName: str, jsonData: json, overwrite=True, va=False) -> Tuple[int, str, str]:
    # input:
    # tosFolder: tos path
    # htmlFileName: file name in Tos
    # jsonData: data for rendering html
    #
    # output:
    # code, msg, resultHtmlLink

    # rendering html
    try:
        htmlContent = url2carouselTemplate.replace("___input_json_data___", jsonData)
    except Exception as e:
        return ErrorCodeRenderError, str(e), None

    # upload to tos
    try:
        url = save_tos(htmlContent.encode("utf-8"), htmlFileName, tosFolder, overwrite=True, va=False)
    except Exception as e:
        return ErrCodeTosError, str(e), None

    return 0, None, url


def renderTextPoster(csv_name):
    import pandas as pd

    df = pd.read_csv(csv_name, on_bad_lines="skip")
    data = []
    for index, row in df.iterrows():
        i = 0
        resImgUrls = []
        to_add = {
            "name": row.get("cid") or row.get("creative_id"),
            "external_url": row.get("external_url"),
        }
        while row.get(f"custom_res_extra_info_{i}") and row.get(f"custom_res_image_url_{i}"):
            if row.get(f"custom_res_image_url_{i}") != "ERROR":
                try:
                    extra_info = eval(row.get(f"custom_res_extra_info_{i}"))
                    if isinstance(extra_info, dict):
                        for key, value in extra_info.items():
                            if key == "to_gen_key_text_infos":
                                text_infos = value
                                if isinstance(text_infos, list):
                                    for j, text_info in enumerate(text_infos):
                                        to_add[f"text_{i}_{key}_{j}"] = json.dumps(text_info, ensure_ascii=False)
                                else:
                                    to_add[f"text_{i}_{key}"] = json.dumps(text_infos, ensure_ascii=False)
                            elif key != "last_html_codes":
                                to_add[f"text_{i}_{key}"] = json.dumps(value)
                    else:
                        to_add[f"text_{i}"] = json.dumps(row.get(f"custom_res_extra_info_{i}"), ensure_ascii=False)
                    resImgUrls.extend(list(json.loads(row.get(f"custom_res_image_url_{i}"))))
                except Exception as e:
                    to_add[f"text_{i}"] = json.dumps(row.get(f"custom_res_extra_info_{i}"), ensure_ascii=False)
                    resImgUrls.extend(list(json.loads(row.get(f"custom_res_image_url_{i}"))))
                    print(f"[renderTextPoster] error: {e}")
            i += 1
        to_add["ImageUrls"] = resImgUrls if len(resImgUrls) < 3 else resImgUrls[-3:]
        data.append(to_add)
        if index > 200:
            break
    print(len(data))
    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    resultUrl = renderUrl2CarouselSingleHtml("text_poster/", csv_name.split("/")[-1][:-4] + f"-{timestamp}.html", json.dumps(data))
    print(f"resultUrl: {resultUrl}")


def renderImageTextPoster(csv_name):
    import pandas as pd

    df = pd.read_csv(csv_name, on_bad_lines="skip")
    data = []
    for index, row in df.iterrows():
        i = 0
        resImgUrls = []
        to_add = {
            "name": index,
            "bg_res_list": row.get("bg_res_list"),
        }
        if row.get("image_with_text_poster_url") and row.get("image_with_text_poster_extra_info"):
            if row.get("image_with_text_poster_url") != "ERROR":
                try:
                    extra_info = eval(row.get("image_with_text_poster_extra_info"))
                    if isinstance(extra_info, dict):
                        for key, value in extra_info.items():
                            if key == "to_gen_key_text_infos":
                                text_infos = value
                                to_add[f"text_{i}_{key}"] = json.dumps(text_infos, ensure_ascii=False)
                            elif key != "last_html_codes":
                                to_add[f"text_{i}_{key}"] = json.dumps(value)
                    else:
                        to_add[f"text_{i}"] = json.dumps(row.get("image_with_text_poster_extra_info"), ensure_ascii=False)
                    resImgUrls.append(row.get("layout_preview_url"))
                    resImgUrls.append(row.get("image_with_text_poster_url"))
                except Exception as e:
                    to_add[f"text_{i}"] = json.dumps(row.get("image_with_text_poster_extra_info"), ensure_ascii=False)
                    resImgUrls.append(row.get("image_with_text_poster_url"))
                    print(f"[renderTextPoster] error: {e}")
            i += 1
        to_add["ImageUrls"] = resImgUrls
        data.append(to_add)
        if index > 300:
            break
    print(len(data))
    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    resultUrl = renderUrl2CarouselSingleHtml("text_poster/", csv_name.split("/")[-1][:-4] + f"-{timestamp}.html", json.dumps(data))
    print(f"resultUrl: {resultUrl}")
