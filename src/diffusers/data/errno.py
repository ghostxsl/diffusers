"""
策略平台平台错误: 50000-50999
输入问题: 40000-40999
策略内部执行错误: 51000-51999
下游依赖错误: 52000-52999
"""

ErrCodeSuccess = 0
ErrCodeEmptyImageInput = 40101
ErrCodeParamError = 40201
ErrCodeNoBackgroundImage = 40401
ErrCodeInternalError = 51000
ErrCodeNoValidHookImage = 51100
ErrCodeCarouselRenderError = 51101
ErrorCodeTextGenerationError = 51102
ErrorCodeEmptyURLUnderstandingOutput = 51103
ErrorCodeURLUnderstandingError = 51104
ErrorCodeHookGenerationError = 51105
ErrorCodeAddCommentBox = 51106
ErrorCodeHookOptimizationError = 51107
ErrorCodeAdsRefreshMusicError = 51108
ErrorCodeImageRankingError = 51109
ErrorCodeGetTargetImageError = 51110
ErrCodeTosError = 51301
ErrCodePortfolioPlatform = 52001
ErrCodeAiCapabilityError = 52002
ErrCodeAigcImageError = 52003
ErrCodeLegoCoreError = 52004
ErrCodeSiteLegoError = 52006
ErrCodeCreativeToolboxError = 52007
ErrCodePainterError = 52008
ErrCodeAigcImagegenError = 52009
ErrCodeSamiError = 52010
ErrCodeVideoarchError = 52011
ErrCodeDataForgeError = 52012
ErrCodeGetURLDataError = 52013
ErrCodeGetImageAttributeError = 52251
ErrorCodeSegmentError = 52401
ErrorCodeOpenaiError = 52402
ErrorCodeBackgenError = 52403
ErrorCodeRenderError = 52501


class WithCodeError(Exception):
    def __init__(self, code, message):
        self.code = code
        self.message = message
        super(WithCodeError, self).__init__(message)

    def get_status_code(self):
        return self.code if self.code else ErrCodeAigcImageError
