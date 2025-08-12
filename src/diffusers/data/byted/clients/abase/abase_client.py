from diffusers.data.byted.clients.abase.abase2 import Abase2ThriftClient

# music_trendy_new_item_cnt
MUSIC_TRENDY_NEW_ITEM_CNT = Abase2ThriftClient(
    consul="bytedance.abase2.ad_sco_creative_video", name_space="ad_sco_creative_video", table_name="music_trendy_new_item_cnt"
)

# music_trendy_item_vv
MUSIC_TRENDY_ITEM_VV = Abase2ThriftClient(
    consul="bytedance.abase2.ad_sco_creative_video", name_space="ad_sco_creative_video", table_name="music_trendy_item_vv"
)

URL_CRAWL_CACHE_CLIENT = Abase2ThriftClient(
    consul="bytedance.abase2.ad_creative_delivery", name_space="ad_creative_delivery", table_name="url_crawl_cache"
)

URL_UNDERSTANDING_CACHE_CLIENT = Abase2ThriftClient(
    consul="bytedance.abase2.ad_creative_delivery", name_space="ad_creative_delivery", table_name="url_understanding_cache"
)

CAROUSEL_GEN_CACHE_CLIENT = Abase2ThriftClient(
    consul="bytedance.abase2.ad_creative_delivery", name_space="ad_creative_delivery", table_name="carousel_gen_cache"
)

if __name__ == "__main__":
    # print(MUSIC_TRENDY_NEW_ITEM_CNT.get("7373224758013935617"))
    # print(MUSIC_TRENDY_ITEM_VV.get("7373224758013935617"))

    import os

    token = os.popen("doas -p ad.creative.image_core_solution printenv SEC_TOKEN_STRING|tail -1 ").read()
    os.environ["SEC_TOKEN_STRING"] = token
    if len(os.getenv("SEC_TOKEN_STRING", "")) < 1:
        token = os.popen("cat /tmp/identity.token").read()
        os.environ["SEC_TOKEN_STRING"] = token
    os.environ["SEC_KV_AUTH"] = "1"
    os.environ["TCE_PSM"] = "ad.creative.image_core_solution"

    print(URL_UNDERSTANDING_CACHE_CLIENT.set("a", "running"))
    print(URL_UNDERSTANDING_CACHE_CLIENT.get("a"))
    print(URL_UNDERSTANDING_CACHE_CLIENT.get("a") == b"running")
