from typing import Callable, Dict, List, Tuple, Any
import concurrent.futures
import logging as logger


def execute_concurrently_mul_func(func_dict: Dict[Callable, List], max_workers: int = 10) -> Dict[str, Tuple]:
    """
    并行执行传入的函数。

    :param func_dict: 一个字典，键为函数，值为包含参数的列表
    :return: 一个字典，键为函数名称，值为一个元组，其中包含传入的参数和函数返回值
    """
    results = {}

    # 定义一个内部函数，用于执行单个任务
    def task(func, args):
        return func(*args)

    # 使用 ThreadPoolExecutor 进行并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = {executor.submit(task, func, args): func for func, args in func_dict.items()}

        # 获取所有结果
        for future in concurrent.futures.as_completed(futures):
            func = futures[future]
            func_name = func.__name__
            try:
                result = future.result()
                results[func_name] = (result, func_dict[func])
            except Exception as exc:
                logger.error(f"[execute_in_parallel] {func_name} generated an exception: {exc}")
                results[func_name] = (None, func_dict[func])

    return results


def execute_concurrently(
    function: Callable[..., Any],
    args_list: List[Tuple],
    max_workers: int,
    timeout: float = 20,
    max_retries: int = 3,
    fail_fast: bool = True,  # raise exception on the first failure
    failure_threshold: float = None,  # e.g., 0.5 = fail if > 50% of tasks fail
    need_log: bool = False,  # 内部组件，有阻塞风险，建议IO密集型任务开启，cpu密集型任务不建议开启
) -> List[Any]:
    results = [None] * len(args_list)
    retry_count = [0] * len(args_list)
    failed_tasks = []

    executor = (
        context.thread_pool_executor.ThreadPoolExecutor(max_workers=max_workers)
        if need_log
        else concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    )

    with executor:
        future_to_index = {executor.submit(function, *args): i for i, args in enumerate(args_list)}

        while future_to_index:
            completed, pending = concurrent.futures.wait(future_to_index, timeout=timeout)

            for future in completed:
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    if retry_count[index] < max_retries:
                        retry_count[index] += 1
                        args = args_list[index]
                        new_future = executor.submit(function, *args)
                        future_to_index[new_future] = index
                        logger.warning(
                            f"[execute_concurrently]【function = {function.__name__}】 retrying task at index {index} (retry {retry_count[index]}) due to exception: {e}"
                        )
                    else:
                        args = args_list[index]
                        logger.error(
                            f"[execute_concurrently]【function = {function.__name__}】 task at index {index} failed after {max_retries} retries, args: {args}, exception: {e}"
                        )
                        failed_tasks.append((index, e))

                        if fail_fast:
                            raise RuntimeError(
                                f"[execute_concurrently]【function = {function.__name__}】 task at index {index} failed after {max_retries} retries, args: {args}"
                            ) from e
                del future_to_index[future]
                del future

    if failure_threshold is not None:
        failure_rate = len(failed_tasks) / len(args_list)
        if failure_rate > failure_threshold:
            raise RuntimeError(
                f"[execute_concurrently]【function = {function.__name__}】 failure rate {failure_rate:.2%} exceeds threshold {failure_threshold:.2%}, failed tasks: {len(failed_tasks)}/{len(args_list)}"
            )

    return results


def add(a, b):
    return a + b


def multiply(a, b):
    return a * b


if __name__ == "__main__":
    # 测试
    functions = {
        add: [1, 2],
        multiply: [2, 3],
    }

    # 并行执行函数并获取结果
    results = execute_concurrently_mul_func(functions)
    print(results)
