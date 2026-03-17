from ..deepseek_ocr import process_vision_info
from functools import partial

process_vision_info_v2 = partial(process_vision_info, add_image_token_id=False)