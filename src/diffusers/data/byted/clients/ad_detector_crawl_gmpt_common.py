import logging

from my_fake_useragent import UserAgent
from overpass_ad_detector_crawl_gmpt_common.clients.rpc.ad_detector_crawl_gmpt_common import AdDetectorCrawl_Gmpt_CommonClient
from overpass_ad_detector_crawl_gmpt_common.euler_gen.ad.crawl_gmpt_common.idl.crawl_thrift import (
    NetworkProxy,
    BizStatusCode,
    RecordCrawlverseRequest,
)


client = AdDetectorCrawl_Gmpt_CommonClient()


def can_download_url(url: str) -> bool:
    logging.info(f"Starting URL validation for: {url}")
    try:
        ua = UserAgent(family="chrome")
        random_agent = ua.random()
        logging.info(f"GeneralUrlCrawlingUserAgency: {random_agent}")

        proxy = NetworkProxy(path="10.8.14.17:8118", user="creative_solution_webhook_tangweichao", password="bsZvwN9r28gTPw")

        validate_req = RecordCrawlverseRequest(business="Creative_Ai_Factory", url=url, ua=random_agent, proxy=proxy)

        code, msg, validate_resp = client.recordCrawlverse(req_object=validate_req)

        if code != 0:
            logging.error(f"Validation service error: {msg}")
            return False

        if validate_resp.code != BizStatusCode.Success:
            logging.error(f"Validation service error: {validate_resp.msg}")
            return False

        if validate_resp.data and not validate_resp.data.is_legal:
            reasons = validate_resp.data.illegal_reasons
            logging.error(f"Url blocked: {url}, reasons: {reasons}")
            return False

        logging.info(f"url_can_download: {url}")
        logging.info(f"return_url: {validate_resp.data.url}")
        return True

    except Exception as e:
        logging.exception(f"URL validation failed: {str(e)}")
        return False
