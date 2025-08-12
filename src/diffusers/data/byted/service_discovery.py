import ipaddress
import random
import logging
from typing import List, Union
from bytedance import servicediscovery
from concurrent.futures import ThreadPoolExecutor, as_completed


def gpu_servicediscovery(
    psm: str,
    cluster: str = "Bernard-Prod",
    idc: Union[str, List[str]] = None,
    env: str = "prod",
    **kwargs,
) -> List[str]:
    """
    GPU服务需要基于服务发现做流量调度，否则无法实现负载均衡
    :param cluster (str): Merlin GPU服务集群名，用于服务发现和流量调度;
    :param psm (str): Merlin GPU服务psm;
    :param idc (Union[str, List[str]]): Merlin GPU服务idc或者idc list;
    :param env (str): Merlin GPU服务env环境;
    :return: 返回为List[str], IP:PORT形式的服务发现结果
    """
    if not idc:
        idc = ["maliva", "sg1", "sg2", "my", "my2", "mya", "myb"]
    elif isinstance(idc, str):
        idc = [idc]
    psm_list = [f"{psm}.service.{v}" for v in idc]

    service_list = []
    with ThreadPoolExecutor(max_workers=len(psm_list)) as executor:
        future_to_psm = {executor.submit(servicediscovery.lookup_name, psm): psm for psm in psm_list}
        for future in as_completed(future_to_psm):
            psm = future_to_psm[future]
            try:
                single_idc_list = future.result()
                for service in single_idc_list:
                    if service.get("Tags", {}).get("cluster") == cluster and service.get("Tags", {}).get("env") == env:
                        service_list.append(service)
            except Exception as e:
                # Handle exceptions raised during service discovery
                logging.error(f"[gpu_servicediscovery] An error occurred for {psm}: {e}")

    def ip_fix(ip_obj: str) -> str:
        # 判断是IPv4还是IPv6地址，IPv6地址需要加上括号
        ip_obj = ipaddress.ip_address(ip_obj)
        if isinstance(ip_obj, ipaddress.IPv4Address):
            return ip_obj
        elif isinstance(ip_obj, ipaddress.IPv6Address):
            return f"[{ip_obj}]"
        else:
            assert False

    endpoints = [f"{ip_fix(item['Host'])}:{item['Port']}" for item in service_list]
    endpoints = list(set(endpoints))
    random.shuffle(endpoints)

    return endpoints


def divide_number(num: int, workers: int) -> List[int]:
    """
    基于服务器实例数量和批处理batch_size大小，算每个实例要处理的batch数量.
    :param num (int): 批处理batch总大小;
    :param workers (int): 服务器实例数量;
    :return: 返回为List[int], 每个实例要处理的batch大小. 例如输入
            例如输入num = 7, workers=5, 返回 [2,2,1,1,1]
    """

    if workers > num:
        return [1] * num
    base_share = num // workers
    remainder = num % workers
    shares = [base_share] * workers
    for i in range(remainder):
        shares[i] += 1
    return shares
