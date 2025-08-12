import os
import thriftpy2
from diffusers.data.git_clone import git_clone

repo_dir = os.path.join(os.path.abspath("."), "idl")

git_clone("https://code.byted.org/idl/base.git", f"{repo_dir}/base")
git_clone("https://code.byted.org/idl/i18n_ad.git", f"{repo_dir}/i18n_ad")
git_clone("https://code.byted.org/idl/ad_comm.git", f"{repo_dir}/ad_comm")

BaseThrift = thriftpy2.load(f"{repo_dir}/base/base.thrift", include_dirs=[".", "..", "/opt/tiger/image_core_solution"])

CreativeImageCoreSolutionThrift = thriftpy2.load(
    f"{repo_dir}/i18n_ad/creative/creative_factory/creative_image_core_solution.thrift",
    include_dirs=[".", "..", "/opt/tiger/image_core_solution"],
)

CreativeFactoryPyThrift = thriftpy2.load(
    f"{repo_dir}/i18n_ad/creative/creative_factory/creative_factory_py.thrift",
    include_dirs=[".", "..", "/opt/tiger/image_core_solution"],
)

CreativeAutoTemplateAlgoThrift = thriftpy2.load(
    f"{repo_dir}/i18n_ad/creative/creative_factory/creative_auto_template_algo.thrift",
    include_dirs=[".", "..", "/opt/tiger/image_core_solution"],
)

CreativeFactoryThrift = thriftpy2.load(
    f"{repo_dir}/i18n_ad/creative/creative_factory/creative_factory.thrift",
    include_dirs=[".", "..", "/opt/tiger/image_core_solution"],
)

# SamiGatewayThrift = thriftpy2.load(
#     "../idl/i18n_ad/creative/creative_factory/audio/sami_gateway.thrift",
#     include_dirs=[".", "..", "/opt/tiger/image_core_solution"],
# )

# metaToDraftStructThrift = thriftpy2.load(
#     "../idl/i18n_ad/smart_creative/meta_to_draft.thrift",
#     include_dirs=[".", "..", "/opt/tiger/image_core_solution"],
# )

# ccDraftStructThrift = thriftpy2.load("../idl/i18n_ad/smart_creative/draft.thrift",
#     include_dirs=[".", "..", "/opt/tiger/image_core_solution"],)

# DraftPostRenderThrift = thriftpy2.load(
#     "../idl/i18n_ad/smart_creative/draft_post_render.thrift",
#     include_dirs=[".", "..", "/opt/tiger/image_core_solution"],
# )

# ccDraftBaseStructThrift = thriftpy2.load(
#     "../idl/i18n_ad/smart_creative/base_param.thrift",
#     include_dirs=[".", "..", "/opt/tiger/image_core_solution"],
# )

GoAICapabilityServiceThrift = thriftpy2.load(
    f"{repo_dir}/i18n_ad/creative/creative_factory/creative_ai_capability.thrift", include_dirs=[".", "..", "/opt/tiger/image_core_solution"]
)

GoImageCapabilityThrift = thriftpy2.load(
    f"{repo_dir}/i18n_ad/creative/creative_factory/capabilities/image_capability.thrift", include_dirs=[".", "..", "/opt/tiger/image_core_solution"]
)


strategyBaseStructThrift = thriftpy2.load(
    f"{repo_dir}/i18n_ad/creative/creative_factory/strategy/capability.thrift",
    include_dirs=[".", "..", "/opt/tiger/image_core_solution"],
)

strategyStructThrift = thriftpy2.load(
    f"{repo_dir}/i18n_ad/creative/creative_factory/strategy/strategy.thrift",
    include_dirs=[".", "..", "/opt/tiger/image_core_solution"],
)

CreativeGatewayThrift = thriftpy2.load(
    f"{repo_dir}/i18n_ad/creative/creative_factory/gateway/gateway.thrift",
    include_dirs=[".", "..", "/opt/tiger/image_core_solution"],
)

LegoCoreThrift = thriftpy2.load(
    f"{repo_dir}/i18n_ad/creative/lego/lego_core.thrift",
    include_dirs=[".", "..", "/opt/tiger/image_core_solution"],
)

CreativeAiCapabilityThrift = thriftpy2.load(
    f"{repo_dir}/i18n_ad/creative/creative_factory/creative_ai_capability.thrift",
    include_dirs=[".", "..", "/opt/tiger/image_core_solution"],
)

AigcImageServiceThrift = thriftpy2.load(
    f"{repo_dir}/i18n_ad/creative/creative_factory/image_generation/aigc_image_service.thrift",
    include_dirs=[".", "..", "/opt/tiger/image_core_solution"],
)

VideoArchGuldanThrift = thriftpy2.load(
    f"{repo_dir}/i18n_ad/videoarch/guldan.thrift",
    include_dirs=[".", "..", "/opt/tiger/image_core_solution"],
)


GenAIImageThrift = thriftpy2.load(
    f"{repo_dir}/i18n_ad/creative/creative_factory/genai/aigc_image.thrift",
    include_dirs=[".", "..", "/opt/tiger/image_core_solution"],
)


Abase2Thrift = thriftpy2.load(
    f"{repo_dir}/i18n_ad/creative/creative_factory/abase2/abase2.thrift",
    include_dirs=[".", "..", "/opt/tiger/image_core_solution"],
)
