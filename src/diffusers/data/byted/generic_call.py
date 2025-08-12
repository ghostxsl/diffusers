#  -*- coding: utf-8 -*-
from __future__ import absolute_import

import base64
import json
import os
import platform
import sys
from dataclasses import dataclass
from typing import Dict, Optional

import lark_oapi as lark
import requests
from bytedenv.idc import DC_SG1
from dataclasses_json import DataClassJsonMixin
from dataclasses_json.core import Json
from retrying import retry

current_dir = os.path.dirname(__file__)

APP_ID = "cli_a3c651c56af9100b"
APP_SECRET = base64.b64decode("bVJXV0dsWElKVXRpZDZKOEYxUlZGZEVPRzUxTk96YjQ=").decode()
client = lark.Client.builder().log_level(lark.LogLevel.DEBUG).app_id(APP_ID).app_secret(APP_SECRET).domain("https://fsopen.bytedance.net").build()

CDN_PREFIX_SG = "https://sf-tk-sg.ibytedtos.com/obj/"
CDN_PREFIX_VA = "https://sf16-muse-va.ibytedtos.com/obj/"


if platform.system() != "Darwin":
    url_endpoint = "https://creative-master-api.byted.org"
else:
    url_endpoint = "http://creative-master-api.tiktok-row.net"


url_generic_call = f"{url_endpoint}/generic_call"


@dataclass
class GenericCallReq(DataClassJsonMixin):
    Psm: str
    Method: str
    RequestJson: str
    Cluster: str = None
    Idc: str = None
    Env: str = None
    ConnectTimeoutMS: int = None
    RpcTimeoutMS: int = None
    NoCache: bool = None

    def format_dict(self, encode_json=False) -> Dict[str, Json]:
        with_none_dict = self.to_dict(encode_json)
        non_none_dict = {k: v for k, v in with_none_dict.items() if v is not None}
        return non_none_dict


@dataclass
class GenericCallData(DataClassJsonMixin):
    code: int
    msg: str
    data: str

    def parse(self) -> Optional[Dict]:
        if not self.data:
            print(f"generic err: {self.msg}, code: {self.code}")
            return None
        try:
            return json.loads(self.data)
        except json.JSONDecoder:
            print(f"invalid generic result: {self.data},msg: {self.msg}, code: {self.code}")
            return None


@retry(
    wait_fixed=250,
    wait_exponential_multiplier=2,
    stop_max_attempt_number=3,
)
def send_request(
    psm: str,
    method: str,
    request_dict: dict,
    cluster: str = "default",
    idc: str = DC_SG1,
    env: str = "prod",
    timeout: int = 60,
) -> dict:
    req = GenericCallReq(
        Psm=psm,
        Method=method,
        RequestJson=json.dumps(request_dict),
        Cluster=cluster,
        Idc=idc,
        Env=env,
        RpcTimeoutMS=timeout * 1000,
    )
    response = requests.post(url=url_generic_call, json=req.format_dict(), timeout=timeout)
    # log_id = response.headers["X-TT-LOGID"]
    # if sys.gettrace():
    #     print(f"{log_id}")
    data = GenericCallData.from_dict(response.json())
    return data.parse()
