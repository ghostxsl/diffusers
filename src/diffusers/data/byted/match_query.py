"""判断图片和文本相匹配"""

import logging
from diffusers.data.byted.decorator import timer, log_io
from diffusers.data.byted.clients.azure_mllm import MLLMClient

gpt_4_1_mini_client = MLLMClient(model_name="gpt-4.1-mini-2025-04-14")

PROMPT = """
You are a permissive title–image compatibility checker.

Goal
Determine whether the given title can be used with at least one image in the input list
without creating a clear contradiction or major confusion. Title: "___title___"

Lenient Policy (bias toward YES)
- Approve unless there is a **clear, specific, and strong** conflict with what is visually shown.
- If unsure, default to **YES**.
- If multiple images are provided, return **YES** if **any one** image is compatible.

What NOT to judge (ignore these; they are not conflicts)
- Real-world feasibility, typical price, plausibility, or likelihood (e.g., “under $1”, discounts).
- Subjective or promotional language (e.g., “best”, “top picks”, “must-have”).
- Missing details not visible in the image (brand, specs, origin, time, stock, warranty).
- General category breadth (e.g., “gadgets”, “summer picks”, “new arrivals”).

Conflicts that DO justify NO (must apply to **all** images)
- Category/object mismatch (e.g., title says “football match” but image shows electronics).
- Explicit textual contradiction visible in the image (e.g., title says “3–0” but the image text shows “1–2”; title claims brand A while the image clearly shows brand B).
- Directional negation that opposes what is shown (e.g., title says “no electronics” while the image is electronics).
- Wrong sport/league/team when those are plainly shown in the image.
- Misleading about the **main subject** (e.g., title says “pets” while image is only laptops).

Evidence Standard
- Base decisions on visible content (including legible on-image text). Do not guess beyond what is shown.
- If text is unreadable/ambiguous, treat it as unknown (not a conflict).

Output (JSON only)
{
  "result": "YES" or "NO",
  "reason": "<short reason if NO, otherwise empty>"
}

Examples for guidance (do not copy to output):
- Title: "Summer Gadget Deals Under $1" + image of headphones/speaker/watch → YES (price plausibility is ignored).
- Title: "Premier League Football Results" + image of Valorant score UI → NO (sport mismatch).
- Title: "New Headphone Launch" + image with several consumer electronics including headphones → YES.
"""


@timer
@log_io
def match_image_with_query(selected_images, title):
    # return True, ""
    prompt = PROMPT.replace("___title___", title)
    try:
        image_list = [imageinfo.URL for imageinfo in selected_images]

        check_result = gpt_4_1_mini_client.make_image_json_request(None, prompt, [], image_list, 10000)
        if check_result["result"] == "YES":
            return True, None

        reason = f"Title: {title}\nImages: {', '.join(image_list)}\nReason: {check_result.get('reason', 'No reason provided')}"

        return False, reason
    except Exception as e:
        logging.error("【match_image_with_query】检查相关性异常 error:{}".format(e))
        return False, str(e)


if __name__ == "__main__":
    is_ok, reason = match_image_with_query(
        ["https://sf16-muse-va.ibytedtos.com/obj/ad-creative/20250828336ecbff91c7a5d74920864a"],
        "Trendy Leather Jacket|Dark denim & silver hardware — your new go-to look!",
    )
    print(is_ok, reason)
