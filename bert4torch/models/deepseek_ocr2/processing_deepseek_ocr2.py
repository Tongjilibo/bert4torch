from ..deepseek_ocr.processing_deepseek_ocr import DeepseekOcrProcessor
from ...processor.processing_utils import register_processor


@register_processor
class DeepseekOcr2Processor(DeepseekOcrProcessor):
    """Processor for DeepSeek-OCR2 model.

    This processor combines image processing and text tokenization for multimodal OCR tasks.
    """

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        super().__init__(image_processor, tokenizer, chat_template=chat_template)
        self.add_image_token = False  # 必须项


__all__ = ["DeepseekOcr2Processor"]
