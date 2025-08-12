import time
import hashlib
import zlib
import dill
from functools import wraps
from typing import Callable, Union
import threading

from contextvars import ContextVar

import logging as logger
from bytedance import metrics

registry_client = metrics.Client(prefix="ad.creative.strategy_public")

""" 初始化计数器 """
request_index: ContextVar[int] = ContextVar("request_index", default=0)
interface_name: ContextVar[str] = ContextVar("interface_name", default="default_name")


def start_metrics(interface_name_str="default_name"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            request_index.set(0)
            interface_name.set(interface_name_str)

            # 调用原始函数
            result = func(*args, **kwargs)
            return result

        return wrapper

    return decorator


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs)
        end_time = time.time()  # 记录结束时间
        execution_time = end_time - start_time
        logger.info(f"[{func.__name__}] cost: {execution_time:.4f} seconds")
        print(f"[{func.__name__}] cost: {execution_time:.4f} seconds")

        request_index.set(request_index.get() + 1)

        tags = {"func_name": func.__name__, "func_order": "%04d" % request_index.get(), "interface_name": interface_name.get()}
        registry_client.emit_timer("function_latency", execution_time, tags)
        registry_client.emit_rate_counter("function_counter", 1, tags)
        return result

    return wrapper


# 错误码上传
def error_code_upload(interface, code, subscene):
    tags = {"error_code": str(code), "interface": interface, "subscene": subscene}
    registry_client.emit_rate_counter("error_code_counter", 1, tags)


def log_io(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 打印函数输入
        logger.info(f"[{func.__name__}] args: {args}, kwargs: {kwargs}")

        # 调用函数并获取其返回值
        result = func(*args, **kwargs)

        # 打印函数输出
        logger.info(f"[{func.__name__}] return: {result}")

        return result

    return wrapper


def cache(
    cli,
    key_func: Union[Callable, str] = None,
    version: str = "v0",
    ttl_days: float = 7,
    timeout: int = 1200,
    interval: int = 1,
    compress: bool = True,
    enable: bool = True,
):
    """
    缓存装饰器，用于缓存函数的返回值。
    :param cli: abase client。位于biz/infra/clients/abase/abase_client.py
    :param key_func: 用于生成缓存键的函数 或者 直接给丁的缓存键值字符串。不构造的时候默认使用装饰函数的所有参数
    :param version: 缓存版本号，用于区分不同版本的缓存。当参数没改变，但是函数逻辑有改变，需要重新缓存的时候，需要增加版本号
    :param ttl_days: 缓存的有效期，单位为天
    :param timeout: 轮询缓存的超时时间，单位为秒。为同一时间请求的长时函数专门设计。为0时不等待其他再请求的函数，直接执行结果
    :param interval: 轮询缓存的间隔时间，单位为秒
    :param compress: 是否压缩缓存值，默认为True
    :param enable: 是否启用缓存，默认为True
    :return: 装饰器函数
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not enable:
                return func(*args, **kwargs)

            # 获取函数名和模块名
            function_name = func.__name__
            module_name = func.__module__

            # Calculate the cache key
            if isinstance(key_func, str):
                raw_key = key_func
            elif callable(key_func):
                raw_key = str(key_func(*args, **kwargs))
            else:
                raw_key = str(args) + str(kwargs)

            # 将模块名、函数名、序列化键、版本和过期时间组合成一个字符串
            full_key = module_name + "." + function_name + "." + raw_key + version + str(ttl_days)

            # Create an MD5 hash of the key
            md5_key = hashlib.md5(full_key.encode("utf-8")).hexdigest()

            # Check the cache
            start_time = time.time()
            while True:
                cached_value = cli.get(md5_key)

                if cached_value is None:
                    logger.info(f"[{module_name}.{function_name}] cache miss, start running.")
                    # Key does not exist, set it to running and execute the function
                    cli.set(md5_key, "running", int(ttl_days * 24 * 60 * 60 * 1000))
                    result = func(*args, **kwargs)
                    encoded_result = dill.dumps(result)
                    if compress:
                        encoded_result = zlib.compress(encoded_result)
                    cli.set(md5_key, encoded_result, ttl_days * 24 * 60 * 60 * 1000)

                    logger.info(f"[{module_name}.{function_name}] cache miss, run and set cache success")

                    return result

                if cached_value != b"running":
                    # Cached value is available and not 'running'
                    if compress:
                        cached_value = zlib.decompress(cached_value)
                    decoded_value = dill.loads(cached_value)
                    logger.info(f"[{module_name}.{function_name}] cache hit, return cache")

                    return decoded_value

                # If it's still running, check if timeout has been reached
                elapsed_time = time.time() - start_time
                if elapsed_time > timeout:
                    # Timeout reached, delete key and run function
                    cli.delete(md5_key)
                    result = func(*args, **kwargs)
                    encoded_result = dill.dumps(result)
                    if compress:
                        encoded_result = zlib.compress(encoded_result)
                    cli.set(md5_key, encoded_result, int(ttl_days * 24 * 60 * 60 * 1000))

                    logger.info(f"[{module_name}.{function_name}] cache timeout, run and reset cache success")

                    return result

                # Wait for the interval before checking again
                time.sleep(interval)

        return wrapper

    return decorator


def timeout(seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [TimeoutError(f'Function "{func.__name__}" timed out after {seconds} seconds')]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    result[0] = e

            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(seconds)
            if thread.is_alive():
                raise TimeoutError(f'Function "{func.__name__}" timed out after {seconds} seconds')
            elif isinstance(result[0], Exception):
                raise result[0]
            else:
                return result[0]

        return wrapper

    return decorator
