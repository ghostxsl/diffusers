# Copy from https://code.byted.org/ad/creative_url2video/blob/master/common/abase2.py
import logging

from diffusers.data.byted.clients.abase.external_thrift import Abase2Thrift
from diffusers.data.byted.middleware import client_logid

import euler
from euler.base_compat_middleware import client_middleware, gdpr_auth_middleware


euler.install_thrift_import_hook()

logger = logging.getLogger(__name__)
logger.setLevel("WARNING")

ThriftService = Abase2Thrift.ThriftService
ErrorCode = Abase2Thrift.ErrorCode

GetRequest = Abase2Thrift.GetRequest
GetResponse = Abase2Thrift.GetResponse
BatchGetRequest = Abase2Thrift.BatchGetRequest
BatchGetResponse = Abase2Thrift.BatchGetResponse

SetRequest = Abase2Thrift.SetRequest
SetResponse = Abase2Thrift.SetResponse
BatchSetRequest = Abase2Thrift.BatchSetRequest
BatchSetResponse = Abase2Thrift.BatchSetResponse

DeleteRequest = Abase2Thrift.DeleteRequest
DeleteResponse = Abase2Thrift.DeleteResponse
BatchDeleteRequest = Abase2Thrift.BatchDeleteRequest
BatchDeleteResponse = Abase2Thrift.BatchDeleteResponse

ExpireRequest = Abase2Thrift.ExpireRequest
ExpireResponse = Abase2Thrift.ExpireResponse
ExpireAtRequest = Abase2Thrift.ExpireAtRequest
ExpireAtResponse = Abase2Thrift.ExpireAtResponse
TtlRequest = Abase2Thrift.TtlRequest
TtlResponse = Abase2Thrift.TtlResponse

_code_to_name = ErrorCode._VALUES_TO_NAMES


class Abase2ThriftClient:
    def __init__(
        self,
        consul,
        name_space,
        table_name,
        timeout=3,
        enable_logging=True,
        **kwargs,
    ):
        """
        Abase2 Thrift Client
        https://bytedance.sg.feishu.cn/wiki/wikcnqSd1cjj683H4GyuMkK6r3c
        """
        if euler.__version__ > "2.0.0":
            self._client: ThriftService = euler.Client(
                ThriftService,
                f"sd://{consul}",  # Abase2服务没有cluster字段 在Euler中需要显式为空
                timeout=timeout,
                without_cluster=True,
                transport='ttheader',
                **kwargs,
            )
        else:
            self._client: ThriftService = euler.Client(
                ThriftService,
                f"sd://{consul}",  # Abase2服务没有cluster字段 在Euler中需要显式为空
                timeout=timeout,
                transport='ttheader',
                **kwargs,
            )
        self._client.use(gdpr_auth_middleware)
        self._client.use(client_middleware)
        self._client.use(client_logid)

        self._name_space = name_space
        self._table_name = table_name

        self._enable_logging = enable_logging

    def get(self, k):
        req = GetRequest(namespace_name=self._name_space, table_name=self._table_name, key=k)
        resp: GetResponse = self._client.Get(req)

        if resp.error_code != ErrorCode.OK:
            v = None
            if self._enable_logging:
                err = _code_to_name[resp.error_code]
                logging.warn(f"Abase2.get key:[{k}] error:[{err}]")
        else:
            v = resp.value

        return v

    def batch_get(self, ks):
        req = BatchGetRequest(
            namespace_name=self._name_space,
            table_name=self._table_name,
            reqs=[GetRequest(key=k) for k in ks],
        )
        resp: BatchGetResponse = self._client.BatchGet(req)

        if resp.error_code != ErrorCode.OK:
            if self._enable_logging:
                err = _code_to_name[resp.error_code]
                logging.warn(f"Abase2.batch_get global error:[{err}]")

        vs = []
        for k, r in zip(ks, resp.resps):
            if r.error_code != ErrorCode.OK:
                v = None
                if self._enable_logging:
                    err = _code_to_name[r.error_code]
                    logging.warn(f"Abase2.batch_get key:[{k}] error:[{err}]")
            else:
                v = r.value
            vs.append(v)

        return vs

    def set(self, k, v, ttl_ms=0):
        req = SetRequest(
            namespace_name=self._name_space,
            table_name=self._table_name,
            key=k,
            value=v,
            ttl_ms=ttl_ms,
        )
        resp: SetResponse = self._client.Set(req)

        is_ok = True
        if resp.error_code != ErrorCode.OK:
            is_ok = False
            if self._enable_logging:
                err = _code_to_name[resp.error_code]
                logging.error(f"Abase2.set key:[{k}] error:[{err}]")

        return is_ok

    def batch_set(self, ks, vs, ttl_ms=0):
        req = BatchSetRequest(
            namespace_name=self._name_space,
            table_name=self._table_name,
            reqs=[SetRequest(key=k, value=v, ttl_ms=ttl_ms) for k, v in zip(ks, vs)],
        )
        resp: BatchSetResponse = self._client.BatchSet(req)

        if resp.error_code != ErrorCode.OK:
            if self._enable_logging:
                err = _code_to_name[resp.error_code]
                logging.error(f"Abase2.batch_set global error:[{err}]")

        is_oks = []
        for k, r in zip(ks, resp.resps):
            is_ok = True
            if r.error_code != ErrorCode.OK:
                is_ok = False
                if self._enable_logging:
                    err = _code_to_name[r.error_code]
                    logging.error(f"Abase2.batch_set key:[{k}] error:[{err}]")
            is_oks.append(is_ok)

        return is_oks

    def delete(self, k):
        req = DeleteRequest(
            namespace_name=self._name_space,
            table_name=self._table_name,
            key=k,
        )
        resp: DeleteResponse = self._client.Delete(req)

        is_ok = True
        if resp.error_code != ErrorCode.OK:
            is_ok = False
            if self._enable_logging:
                err = _code_to_name[resp.error_code]
                logging.error(f"Abase2.delete key:[{k}] error:[{err}]")

        return is_ok

    def batch_delete(self, ks):
        req = BatchDeleteRequest(
            namespace_name=self._name_space,
            table_name=self._table_name,
            reqs=[DeleteRequest(key=k) for k in ks],
        )
        resp: BatchDeleteResponse = self._client.BatchDelete(req)

        if resp.error_code != ErrorCode.OK:
            if self._enable_logging:
                err = _code_to_name[resp.error_code]
                logging.error(f"Abase2.batch_delete global error:[{err}]")

        is_oks = []
        for k, r in zip(ks, resp.resps):
            is_ok = True
            if r.error_code != ErrorCode.OK:
                is_ok = False
                if self._enable_logging:
                    err = _code_to_name[r.error_code]
                    logging.error(f"Abase2.batch_delete key:[{k}] error:[{err}]")
            is_oks.append(is_ok)

        return is_oks

    def expire(self, k, ttl_ms):
        req = ExpireRequest(
            namespace_name=self._name_space,
            table_name=self._table_name,
            key=k,
            ttl_ms=ttl_ms,
        )
        resp: ExpireResponse = self._client.Expire(req)

        is_ok = True
        if resp.error_code != ErrorCode.OK:
            is_ok = False
            if self._enable_logging:
                err = _code_to_name[resp.error_code]
                logging.error(f"Abase2.expire key:[{k}] error:[{err}]")

        return is_ok

    def expire_at(self, k, expireat_ms=0):
        req = ExpireAtRequest(
            namespace_name=self._name_space,
            table_name=self._table_name,
            key=k,
            expireat_ms=expireat_ms,
        )
        resp: ExpireAtResponse = self._client.ExpireAt(req)

        is_ok = True
        if resp.error_code != ErrorCode.OK:
            is_ok = False
            if self._enable_logging:
                err = _code_to_name[resp.error_code]
                logging.error(f"Abase2.expire_at key:[{k}] error:[{err}]")

        return is_ok

    def ttl_ms(self, k):
        req = TtlRequest(
            namespace_name=self._name_space,
            table_name=self._table_name,
            key=k,
        )
        resp: TtlResponse = self._client.Ttl(req)

        if resp.error_code != ErrorCode.OK:
            ttl = 0
            if self._enable_logging:
                err = _code_to_name[resp.error_code]
                logging.error(f"Abase2.ttl_ms key:[{k}] error:[{err}]")
        else:
            ttl = resp.ttl_ms

        return ttl


if __name__ == "__main__":
    # pass
    import json

    key = "web_or_app@web_ecom"
    # key = "total_fused_template_ids"
    cc4b_template_client = Abase2ThriftClient(
        consul="bytedance.abase2.ad_sco_creative_video",
        name_space="ad_sco_creative_video",
        # table_name="cc_script_template_video_material_info"
        table_name="cc4b_template",
    )
    res = json.loads(cc4b_template_client.get(key))
    print(res)
    print(len(res))
    print(len(list(set(res))))
