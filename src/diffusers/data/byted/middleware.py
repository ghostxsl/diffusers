import json
import time
import traceback
import os
import logging

import byteddps
import euler
from euler import base_compat_middleware
from euler.context import ReadOnlyDict

from diffusers.data.byted.errno import WithCodeError
from overpass_ad_creative_image_core_solution.euler_gen.idl.i18n_ad.creative.creative_factory import (
    creative_image_core_solution_thrift as service_main_thrift,
)
from overpass_ad_creative_image_core_solution.euler_gen.idl.base.base_thrift import BaseResp, Base


#################### server level ####################
def panic_middleware(ctx, *args, **kwargs):
    method = ctx.local.get("method")
    try:
        return ctx.next(*args, **kwargs)
    except Exception as e:
        logging.error(f"Execution err: {e}\n track:{traceback.format_exc()}")
        try:
            basic_func_resp = getattr(service_main_thrift, method + "Resp")
            statusCode = 50000
            if getattr(basic_func_resp, "BaseResp", None) is not None:
                statusCode = getattr(basic_func_resp.BaseResp, "StatusCode", 50000)
            elif isinstance(e, WithCodeError) or isinstance(e, RPCError):
                statusCode = e.code
            logging.info(f"basic_func_resp: {basic_func_resp}")
            setattr(basic_func_resp, "BaseResp", BaseResp(StatusCode=statusCode, StatusMessage=f"{e}"))
            logging.info(f"basic_func_resp: {basic_func_resp}")
            return basic_func_resp
        except Exception as middleware_e:
            logging.error(f"panic_middleware for method: {method}, middleware_err: {middleware_e}, err: {e}")
            return  ## {"BaseResp": {"StatusCode": 50000, "StatusMessage": f"{e}"}}


def print_req_middleware(ctx, *args, **kwargs):
    method = ctx.local.get("method")
    logging.info(f"Call {method}..., req: {args}")
    return ctx.next(*args, **kwargs)


class RPCError(Exception):
    def __init__(self, code, message):
        self.code = code
        self.message = message
        super(RPCError, self).__init__(message)


def check_base_resp_middleware(ctx, *args, **kwargs):
    resp = ctx.next(*args, **kwargs)
    if resp.BaseResp and resp.BaseResp.StatusCode != 0:
        raise RPCError(resp.BaseResp.StatusCode, resp.BaseResp.StatusMessage)
    return resp


def _extract_context(args, ctx):
    if not args:
        return []
    req = args[0]

    if not hasattr(req, "Base"):
        return

    if not hasattr(req.Base, "extra") or not req.Base.extra:
        return

    tmp_transient = {}

    if ctx.transient:
        tmp_transient.update(ctx.transient)

    user_extra = {}
    user_extra_raw = req.Base.extra.get("user_extra")
    if user_extra_raw:
        val = None

        try:
            val = json.loads(user_extra_raw)
        except ValueError as exc:
            raise ValueError("invalid type for Base.Extra.user_extra") from exc

        if isinstance(val, dict):
            user_extra.update(val)
        else:
            raise ValueError("invalid type for Base.Extra.user_extra")

    if user_extra:
        for key, value in user_extra.items():
            if key.startswith("RPC_TRANSIT_"):
                if not tmp_transient.get(key[len("RPC_TRANSIT_") :]):
                    tmp_transient[key[len("RPC_TRANSIT_") :]] = value

    ctx._transient = ReadOnlyDict(tmp_transient)


def extra_middleware(ctx, *args, **kwargs):
    _extract_context(args, ctx)
    result = ctx.next(*args, **kwargs)
    return result


#################### client level ####################
def get_euler_client(service_cls, target, **kwargs):
    client = euler.Client(service_cls, target, disable_logging=False, transport='ttheader', **kwargs)
    client.use(base_compat_middleware.gdpr_auth_middleware)
    client.use(client_logid)
    client.use(base_compat_middleware.client_middleware)
    # client.use(env_middleware)
    return client


def set_token_middleware(ctx, *args, **kwargs):
    logging.info("in set_token_middleware")
    if os.getenv("TCE_HOST_ENV", "notTCE") == "notTCE":
        logging.info("not TCE ENV, set token middleware")
        base = getattr(args[0], "Base", Base())
        psm = os.environ.get("TCE_PSM")
        token = byteddps.get_token()[:-1]
        if psm is None or token is None:
            logging.error(f"psm or token is None, psm: {psm}, token: {token}")
        else:
            base.Caller = psm
            base.Extra = {"gdpr-token": token}
            setattr(args[0], "Base", base)
    return ctx.next(*args, **kwargs)


def client_logid(ctx, *args, **kwargs):
    try:
        return ctx.next(*args, **kwargs)
    finally:
        logid = ctx.persistent.get("logid")
        method = ctx.local.get("method", "-")
        logging.info(f"Call {method}..., logid: {logid}")


# if needed
def env_middleware(ctx, *args, **kwargs):
    ppe_env = "ppe_cue_strategy_platform"
    ctx.persistent["env"] = ppe_env
    return ctx.next(*args, **kwargs)


def calc_runtime_middleware(ctx, *args, **kwargs):
    start = time.time()
    result = ctx.next(*args, **kwargs)
    end = time.time()
    logging.info(f"calc_runtime_middleware: {end - start}")
    if end - start > 120:
        logging.info(f"Request timed too long, cost{end - start} seconds.")
    return result


if __name__ == "__main__":
    try:
        result = getattr(service_main_thrift, "AutoTemplatePreProcessingResp")
        setattr(result, "BaseResp", {"StatusCode": 500, "StatusMessage": "111"})
        print(result.BaseResp)
    except Exception as e:
        print(e)
