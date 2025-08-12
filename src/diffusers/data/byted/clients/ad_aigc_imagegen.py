import logging
import traceback
import bytedlogger
from copy import deepcopy
from typing import Union, Optional, List
from concurrent.futures import as_completed
from diffusers.data.byted.middleware import get_euler_client
from bytedance.context.thread_pool_executor import ThreadPoolExecutor
from diffusers.data.byted.service_discovery import gpu_servicediscovery, divide_number
from diffusers.data.byted.clients.abase.external_thrift import GenAIImageThrift, BaseThrift
import diffusers.data.byted.errno as err

bytedlogger.config_default()

Fluxs_PSM_DEFAULT = "ad.aigc.dynamic_img_gen"
FluxsBackgenCli_CLUSTER_DEFAULT = "flux-s-dynamic-backgen"

Fluxs_Adapter_PSM_DEFAULT = "ad.aigc.imagegen"
FluxsAdapterBackgenCli_CLUSTER_DEFAULT = "flux-adapter-backgen"

Seed_PSM_DEFAULT = "ad.aigc.dynamic_img_gen"
SeedBackgenCli_CLUSTER_DEFAULT = "seed-backgen"

Fluxs_TIMEOUT = 120
Seed_TIMEOUT = 300

CalSolidColorCli = get_euler_client(GenAIImageThrift.AIGCImageService, "sd://ad.aigc.imagegen?idc=my&cluster=calcalate_solid_color", timeout=60)

CalColorCli = get_euler_client(GenAIImageThrift.AIGCImageService, "sd://ad.aigc.imagegen?idc=maliva&cluster=calculate_color", timeout=20)


def backgen_genai_fluxs(
    image: Union[str, bytes],
    ref_image: Optional[Union[str, bytes]] = None,
    inpaint_mask: Optional[Union[str, bytes]] = None,
    prompt: Union[List[str], str] = "",
    num: int = 1,
    cluster: str = None,
    trade_gpt4o="i2v",
    **kwargs,
) -> dict:
    """
    genai的背景生成服务flux.s版本
    https://bytedance.larkoffice.com/docx/Drgvdel02oJhIbxtDhbcnWsqndg
    :param image: 输入图片url或者bytes;
    :param ref_image: 参考图片url或者bytes，可以为空，用于预设场景或者参考图出图;
    :param inpaint_mask: mask图片url或者bytes，为空时则会直接走服务端默认抠图;
    :param prompt: 用户输入的prompt, 可以为空; 可以为List，但需要跟num对齐;
    :param num: 一次生成num张图片;
    :param cluster: GenAI生图服务集群名，用于服务发现和流量调度;
    :param trade_gpt4o: i2v/non-i2v，i2v/抽象换背景
    :return: 返回为dict
        image_bytes_list: 结果图片bytes list
        prompt: 实际使用的prompt list
    """
    if not cluster:
        ti2_cluster = FluxsBackgenCli_CLUSTER_DEFAULT
        i2i_cluster = FluxsAdapterBackgenCli_CLUSTER_DEFAULT

    if isinstance(prompt, str):
        prompt = [prompt] * num
    elif prompt is None:
        prompt = [""] * num
    elif isinstance(prompt, list):
        if len(prompt) != num:
            raise ValueError("The length of prompt list must be equal to num.")

    req = GenAIImageThrift.FluxBackGenRequest(
        image=GenAIImageThrift.ImageInfo(image_url=None, image_data=None),
        inpaint_mask=None,
        prompt=prompt[0] or "",  # use self prompt
        use_self_ref_image=GenAIImageThrift.ImageInfo(image_url=None, image_data=None),  # use reference img
        num_inference_steps=15,  # infer step
        num_images_per_prompt=num,
        return_type=GenAIImageThrift.ImageReturnType.Binary,  # ImageReturnType.Tos, #Binary
        # process_image_width=768,
        # process_image_height=1280,
        seed=None,
        need_resize_product=0,
        erude_size_in_mask=-1,
        # you can test different values to get different seg_mask results with different erude. [Note]: erude_size_in_mask<0 uses our default erude strategy, if you don't want to erude, you can set erude_size_in_mask=0
        trade_gpt4o=trade_gpt4o,
        Base=BaseThrift.Base(),
    )

    if ref_image:
        req.adapter_scale = 0.7
        endpoints = gpu_servicediscovery(Fluxs_Adapter_PSM_DEFAULT, i2i_cluster)
    else:
        endpoints = gpu_servicediscovery(Fluxs_PSM_DEFAULT, ti2_cluster)
    cli_list = [get_euler_client(GenAIImageThrift.ImageService, f"tcp://{endpoint}", timeout=Fluxs_TIMEOUT) for endpoint in endpoints]

    if isinstance(ref_image, str):
        req.use_self_ref_image = GenAIImageThrift.ImageInfo(image_url=ref_image, image_data=None)
    elif isinstance(ref_image, bytes):
        req.use_self_ref_image = GenAIImageThrift.ImageInfo(image_url=None, image_data=ref_image)

    if isinstance(image, str):
        req.image = GenAIImageThrift.ImageInfo(image_url=image, image_data=None)
    elif isinstance(image, bytes):
        req.image = GenAIImageThrift.ImageInfo(image_url=None, image_data=image)

    if isinstance(inpaint_mask, str):
        req.inpaint_mask = GenAIImageThrift.ImageInfo(image_url=inpaint_mask, image_data=None)
    elif isinstance(inpaint_mask, bytes):
        req.inpaint_mask = GenAIImageThrift.ImageInfo(image_url=None, image_data=inpaint_mask)

    image_bytes_list = []
    res_prompt = []
    # 流量调度，多进程并发请求，充分利用集群资源
    with ThreadPoolExecutor(max_workers=min(num, len(cli_list), 10)) as executor:
        futures = {}
        # 如果是同一个prompt，则一个请求内num可以大于1
        if len(set(prompt)) == 1:
            executor_num_list = divide_number(num, len(cli_list))
        else:
            executor_num_list = [1] * num
        for i, executor_num in enumerate(executor_num_list):
            executor_req = deepcopy(req)
            executor_req.num_images_per_prompt = executor_num
            executor_req.prompt = prompt[i]
            futures[executor.submit(cli_list[i % len(cli_list)].FluxBackGen, executor_req)] = i
        index_list = []
        for future in as_completed(futures):
            index = futures[future]
            index_list.append(index)
            executor_num = executor_num_list[index]
            try:
                result = future.result()
                if result.BaseResp.StatusCode != 0:
                    raise Exception(f"[backgen_genai flux.s] failed: {result.BaseResp.StatusCode}, {result.BaseResp.StatusMessage}")
                image_bytes_list.extend([img_info.image_data for img_info in result.images])
                res_prompt.extend([result.use_prompt] * executor_num)
            except Exception as e:
                image_bytes_list.extend([None] * executor_num)
                res_prompt.extend([None] * executor_num)
                traceback.print_exc()
                logging.warning(f"[backgen_genai flux.s] An error occurred while batch infer: {e}. Client: {endpoints[index]}")
                continue

    # 保序
    image_bytes_list = [x for _, x in sorted(zip(index_list, image_bytes_list))]
    res_prompt = [x for _, x in sorted(zip(index_list, res_prompt))]

    if all(image_bytes is None for image_bytes in image_bytes_list):
        raise Exception("[backgen_genai flux.s] batch infer failed.")

    res = {"image_bytes_list": image_bytes_list, "prompt": res_prompt}

    logging.info(f"[backgen_genai flux.s] success. {res_prompt}")

    return res


def backgen_genai_seed(
    image: Union[str, bytes],
    inpaint_mask: Optional[Union[str, bytes]] = None,
    prompt: Union[List[str], str] = "",
    num: int = 1,
    cluster: str = None,
    trade_gpt4o="i2v",
    **kwargs,
) -> dict:
    """
    genai的背景生成服务seed版本
    https://bytedance.larkoffice.com/docx/Drgvdel02oJhIbxtDhbcnWsqndg
    :param image: 输入图片url或者bytes;
    :param inpaint_mask: mask图片url或者bytes，为空时则会直接走服务端默认抠图;
    :param prompt: 用户输入的prompt, 可以为空; 可以为List，但需要跟num对齐;
    :param num: 一次生成num张图片;
    :param cluster: GenAI生图服务集群名，用于服务发现和流量调度;
    :param trade_gpt4o: i2v/non-i2v，i2v/抽象换背景
    :return: 返回为dict
        image_bytes_list: 结果图片bytes list
        prompt: 实际使用的prompt list
    """
    if not cluster:
        ti2_cluster = SeedBackgenCli_CLUSTER_DEFAULT

    if isinstance(prompt, str):
        prompt = [prompt] * num
    elif prompt is None:
        prompt = [""] * num
    elif isinstance(prompt, list):
        if len(prompt) != num:
            raise ValueError("The length of prompt list must be equal to num.")

    req = GenAIImageThrift.SeedBackGenRequest(
        image=GenAIImageThrift.ImageInfo(image_url=None, image_data=None),
        inpaint_mask=None,
        prompt=prompt[0] or "",  # use self prompt
        num_inference_steps=12,  # infer step
        num_images_per_prompt=num,
        return_type=GenAIImageThrift.ImageReturnType.Binary,  # ImageReturnType.Tos, #Binary
        # process_image_width=768,
        # process_image_height=1280,
        seed=None,
        need_resize_product=0,
        erude_size_in_mask=-1,  # you can test different values to get different seg_mask results with different erude. [Note]: erude_size_in_mask<0 uses our default erude strategy, if you don't want to erude, you can set erude_size_in_mask=0
        need_super_resolution=1,
        binary_mask=0,
        trade_gpt4o=trade_gpt4o,
        gen_ratio=0,  # 0 ourput image resulution is the same as input; width/height
        Base=BaseThrift.Base(),
    )

    endpoints = gpu_servicediscovery(Seed_PSM_DEFAULT, ti2_cluster)
    cli_list = [get_euler_client(GenAIImageThrift.ImageService, f"tcp://{endpoint}", timeout=Seed_TIMEOUT) for endpoint in endpoints]

    if isinstance(image, str):
        req.image = GenAIImageThrift.ImageInfo(image_url=image, image_data=None)
    elif isinstance(image, bytes):
        req.image = GenAIImageThrift.ImageInfo(image_url=None, image_data=image)

    if isinstance(inpaint_mask, str):
        req.inpaint_mask = GenAIImageThrift.ImageInfo(image_url=inpaint_mask, image_data=None)
    elif isinstance(inpaint_mask, bytes):
        req.inpaint_mask = GenAIImageThrift.ImageInfo(image_url=None, image_data=inpaint_mask)

    image_bytes_list = []
    res_prompt = []
    # 流量调度，多进程并发请求，充分利用集群资源
    with ThreadPoolExecutor(max_workers=min(num, len(cli_list), 10)) as executor:
        futures = {}
        # 如果是同一个prompt，则一个请求内num可以大于1
        if len(set(prompt)) == 1:
            executor_num_list = divide_number(num, len(cli_list))
        else:
            executor_num_list = [1] * num
        for i, executor_num in enumerate(executor_num_list):
            executor_req = deepcopy(req)
            executor_req.num_images_per_prompt = executor_num
            executor_req.prompt = prompt[i]
            futures[executor.submit(cli_list[i % len(cli_list)].SeedBackGen, executor_req)] = i
        index_list = []
        for future in as_completed(futures):
            index = futures[future]
            index_list.append(index)
            executor_num = executor_num_list[index]
            try:
                result = future.result()
                if result.BaseResp.StatusCode != 0:
                    raise Exception(f"[backgen_genai seed] failed: {result.BaseResp.StatusCode}, {result.BaseResp.StatusMessage}")
                image_bytes_list.extend([img_info.image_data for img_info in result.images])
                res_prompt.extend([result.use_prompt] * executor_num)
            except Exception as e:
                image_bytes_list.extend([None] * executor_num)
                res_prompt.extend([None] * executor_num)
                traceback.print_exc()
                logging.warning(f"[backgen_genai seed] An error occurred while batch infer: {e}. Client: {endpoints[index]}")
                continue

    # 保序
    image_bytes_list = [x for _, x in sorted(zip(index_list, image_bytes_list))]
    res_prompt = [x for _, x in sorted(zip(index_list, res_prompt))]

    if all(image_bytes is None for image_bytes in image_bytes_list):
        raise Exception("[backgen_genai seed] batch infer failed.")

    res = {"image_bytes_list": image_bytes_list, "prompt": res_prompt}

    logging.info(f"[backgen_genai seed] success. {res_prompt}")

    return res



def backgen_genai_seed_multi_image(
    image: List[Union[str, bytes]],
    inpaint_mask: Optional[List[Union[str, bytes]]] = None,
    prompt: Union[List[str], str] = "",
    cluster: str = None,
    **kwargs,
) -> dict:
    """
    genai的背景生成服务seed版本
    https://bytedance.larkoffice.com/docx/Drgvdel02oJhIbxtDhbcnWsqndg
    :param image: 输入图片url或者bytes list;
    :param inpaint_mask: mask图片url或者bytes list，为空时则会直接走服务端默认抠图;
    :param prompt: 用户输入的prompt, 可以为空; 可以为List，但需要跟image长度对齐;
    :param cluster: GenAI生图服务集群名，用于服务发现和流量调度;
    :return: 返回为dict
        image_bytes_list: 结果图片bytes list
        prompt: 实际使用的prompt list
    """
    if not cluster:
        ti2_cluster = SeedBackgenCli_CLUSTER_DEFAULT

    if isinstance(prompt, str):
        prompt = [prompt] * len(image)
    elif prompt is None:
        prompt = [""] * len(image)
    elif isinstance(prompt, list):
        if len(prompt) != len(image):
            raise ValueError("The length of prompt list must be equal to image list.")

    if isinstance(inpaint_mask, list):
        if len(inpaint_mask)!= len(image):
            raise ValueError("The length of inpaint_mask list must be equal to image list.")


    req_list = []
    num = len(image)
    for i in range(num):
        req = GenAIImageThrift.SeedBackGenRequest(
            image=GenAIImageThrift.ImageInfo(image_url=None, image_data=None),
            inpaint_mask=None,
            prompt=prompt[i] or "",  # use self prompt
            num_inference_steps=12,  # infer step
            num_images_per_prompt=1,
            return_type=GenAIImageThrift.ImageReturnType.Binary,  # ImageReturnType.Tos, #Binary
            # process_image_width=768,
            # process_image_height=1280,
            seed=None,
            need_resize_product=0,
            erude_size_in_mask=-1,  # you can test different values to get different seg_mask results with different erude. [Note]: erude_size_in_mask<0 uses our default erude strategy, if you don't want to erude, you can set erude_size_in_mask=0
            need_super_resolution=1,
            binary_mask=0,
            trade_gpt4o='i2v',
            gen_ratio=0,  # 0 ourput image resulution is the same as input; width/height
            Base=BaseThrift.Base(),
        )

        if isinstance(image[i], str):
            req.image = GenAIImageThrift.ImageInfo(image_url=image[i], image_data=None)
        elif isinstance(image[i], bytes):
            req.image = GenAIImageThrift.ImageInfo(image_url=None, image_data=image[i])

        if isinstance(inpaint_mask, list):
            if isinstance(inpaint_mask[i], str):
                req.inpaint_mask = GenAIImageThrift.ImageInfo(image_url=inpaint_mask[i], image_data=None)
            elif isinstance(inpaint_mask[i], bytes):
                req.inpaint_mask = GenAIImageThrift.ImageInfo(image_url=None, image_data=inpaint_mask[i])
        req_list.append(req)
    endpoints = gpu_servicediscovery(Seed_PSM_DEFAULT, ti2_cluster)
    cli_list = [get_euler_client(GenAIImageThrift.ImageService, f"tcp://{endpoint}", timeout=Seed_TIMEOUT) for endpoint in endpoints]

    image_bytes_list = []
    res_prompt = []
    # 流量调度，多进程并发请求，充分利用集群资源
    with ThreadPoolExecutor(max_workers=min(num, len(cli_list), 10)) as executor:
        futures = {}
        executor_num_list = [1] * num
        for i, executor_num in enumerate(executor_num_list):
            executor_req = req_list[i]
            executor_req.num_images_per_prompt = executor_num
            executor_req.prompt = prompt[i]
            futures[executor.submit(cli_list[i % len(cli_list)].SeedBackGen, executor_req)] = i
        index_list = []
        for future in as_completed(futures):
            index = futures[future]
            index_list.append(index)
            executor_num = executor_num_list[index]
            try:
                result = future.result()
                if result.BaseResp.StatusCode != 0:
                    raise Exception(f"[backgen_genai seed] failed: {result.BaseResp.StatusCode}, {result.BaseResp.StatusMessage}")
                image_bytes_list.extend([img_info.image_data for img_info in result.images])
                res_prompt.extend([result.use_prompt] * executor_num)
            except Exception as e:
                image_bytes_list.extend([None] * executor_num)
                res_prompt.extend([None] * executor_num)
                traceback.print_exc()
                logging.warning(f"[backgen_genai seed] An error occurred while batch infer: {e}. Client: {endpoints[index]}")
                continue

    # 保序
    image_bytes_list = [x for _, x in sorted(zip(index_list, image_bytes_list))]
    res_prompt = [x for _, x in sorted(zip(index_list, res_prompt))]

    if all(image_bytes is None for image_bytes in image_bytes_list):
        raise Exception("[backgen_genai seed] batch infer failed.")

    res = {"image_bytes_list": image_bytes_list, "prompt": res_prompt}

    logging.info(f"[backgen_genai seed] success. {res_prompt}")

    return res


def calculate_solid_color(
    image_url: str,
    image_bytes: bytes = None,
    mask_url: str = None,
    mask_bytes: bytes = None,
    merge_tolerance: int = 10,
    light_addition: int = 0,
    threshold: int = 1.0,
) -> GenAIImageThrift.CalculateSolidColorResponse:
    # https://bytedance.larkoffice.com/docx/XmcDdTYzqomYDQxeMuncLhrpnwf

    req = GenAIImageThrift.CalculateSolidColorRequest(
        image=GenAIImageThrift.ImageInfo(image_url=image_url, image_data=image_bytes),
        merge_Tolerance=merge_tolerance,
        light_Addition=light_addition,
        threshold=threshold,
        mask=GenAIImageThrift.ImageInfo(image_url=mask_url, image_data=mask_bytes),
    )

    resp = CalSolidColorCli.CalculateSolidColor(req)

    if resp.BaseResp.StatusCode != 0:
        error_msg = f"[calculate_solid_color] failed, code: {resp.BaseResp.StatusCode}, msg: {resp.BaseResp.StatusMessage}"
        logging.error(error_msg)
        raise err.WithCodeError(err.ErrCodeAigcImagegenError, error_msg)

    return resp


def calculate_color(
    image_url: str, image_data: bytes = None, mask_url: str = None, mask_data: bytes = None, trade: str = None
) -> GenAIImageThrift.CalculateColorResponse:
    # https://bytedance.larkoffice.com/docx/HZeYdtN20oYrLaxPvrIc4fxlnqc

    req = GenAIImageThrift.CalculateColorRequest(
        image_url=image_url,
        image_data=image_data,
        mask_url=mask_url,
        mask_data=mask_data,
        trade=trade,
    )
    resp = CalColorCli.CalculateColor(req)

    if resp.BaseResp.StatusCode != 0:
        error_msg = f"[calculate_color] failed, code: {resp.BaseResp.StatusCode}, msg: {resp.BaseResp.StatusMessage}"
        logging.error(error_msg)
        raise err.WithCodeError(err.ErrCodeAigcImagegenError, error_msg)

    return resp


if __name__ == "__main__":
    # # 背景生成
    # # prompt-free/ 抽象换背景
    from time import time
    from biz.infra.clients.tos import save_tos, make_key

    # url = "https://p16-oec-ttp.tiktokcdn-us.com/tos-useast5-i-omjb5zjo8w-tx/ae8bc0d5b33e40f8ac99f5c57b36d9b8~tplv-omjb5zjo8w-resize-jpeg:1000:1000.image?"
    # url = "https://p19-creative-tool-sg.ibyteimg.com/tos-alisg-i-n2703mo9gi-sg/bc72d6f61275403ca265dc79f3641b40~tplv-n2703mo9gi-webp:1280:1280.image"
    # url = "https://sf16-muse-va.ibytedtos.com/obj/ad-creative/20250208336e560f433ac10040dbaeca"
    # mask_url = "https://sf16-muse-va.ibytedtos.com/obj/ad-creative/20250208336e9be551ea113c4c11b3ea"
    # s = time()
    # # res = backgen_genai_fluxs(image=url, num=7, prompt=['1', '2','3','4','5','6','7'])
    # # res = backgen_genai_fluxs(image=url, num=1, trade_gpt4o="non-i2v")
    # res = backgen_genai_seed(image=url, num=1, trade_gpt4o="non-i2v")
    # img_bytes = res["image_bytes_list"][0]
    # res_url = save_tos(img_bytes, make_key())
    # e = time()
    # print("time cost: ", e - s)
    # print("res_url: ", res_url)


    # 背景生成
    # multi-image
    url = "https://p16-oec-ttp.tiktokcdn-us.com/tos-useast5-i-omjb5zjo8w-tx/ae8bc0d5b33e40f8ac99f5c57b36d9b8~tplv-omjb5zjo8w-resize-jpeg:1000:1000.image?"
    url = "https://p19-creative-tool-sg.ibyteimg.com/tos-alisg-i-n2703mo9gi-sg/bc72d6f61275403ca265dc79f3641b40~tplv-n2703mo9gi-webp:1280:1280.image"
    url = "https://sf16-muse-va.ibytedtos.com/obj/ad-creative/20250208336e560f433ac10040dbaeca"
    mask_url = "https://sf16-muse-va.ibytedtos.com/obj/ad-creative/20250208336e9be551ea113c4c11b3ea"
    s = time()
    res = backgen_genai_seed_multi_image(image=[url]*2, inpaint_mask=[mask_url]*2, prompt='test')
    img_bytes_list = [v for v in res["image_bytes_list"]]
    res_url_list = [save_tos(img_bytes, make_key()) for img_bytes in img_bytes_list]
    e = time()
    print("time cost: ", e - s)
    print("res_url: ", res_url_list)


    # # # ref-image based
    # # url = "https://p16-oec-ttp.tiktokcdn-us.com/tos-useast5-i-omjb5zjo8w-tx/ae8bc0d5b33e40f8ac99f5c57b36d9b8~tplv-omjb5zjo8w-resize-jpeg:1000:1000.image?"
    # # ref_url = "http://lf9-lkcdn-tos.byted-ug.com/obj/adsources-console/faadeca0df7dee464486da65b99fa72c"
    # # s = time()
    # # res = backgen_genai_fluxs(image=url, ref_image=ref_url, num=1)
    # # img_bytes = res["image_bytes_list"][0]
    # # img_pil = bytes_to_image(img_bytes)
    # # e = time()
    # # print("time cost: ", e - s)
    # # img_pil.save("backgen_genai_fluxs_ref-image.png")

    # # 算色

    # url = "https://p16-creative-tool-sg.tiktokcdn.com/tos-alisg-i-n2703mo9gi-sg/fb89ba397ce742d2af7aa9a593ff60e7~tplv-n2703mo9gi-webp:816:816.webp"
    # url = "https://lf9-lkcdn-tos.byted-ug.com/obj/adsources-console/faadeca0df7dee464486da65b99fa72c"
    # print(calculate_solid_color(url))  # 是否纯色背景
    # # print(calculate_color(image_url=None, image_data=url_to_bytes(url), trade="vsa"))  # 算色
