from ...processor.feature_extraction_utils import BatchFeature
from ...processor.image_utils import ImageInput
from ...processor.processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack, register_processor
from ...tokenizers.tokenization_utils_base import PreTokenizedInput, TextInput
from ...snippets import auto_docstring
import torch
import math


class DeepseekOcrProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_mm_token_type_ids": True,
        },
        "images_kwargs": {
            "crop_mode": True,
            "base_size": 1024,
            "crop_thread": 768,
            "image_size": 768,
            "dynamic_preprocess_max_num": 6,
        },
    }


def text_encode(tokenizer, text: str, bos: bool = True, eos: bool = False):
    """Encode text to token ids."""
    t = tokenizer.encode(text, add_special_tokens=False)
    bos_id = 0
    eos_id = 1
    if bos:
        t = [bos_id] + t
    if eos:
        t = t + [eos_id]
    return t


@auto_docstring
@register_processor
class DeepseekOcrProcessor(ProcessorMixin):
    """Processor for DeepSeek-OCR model.

    This processor combines image processing and text tokenization for multimodal OCR tasks.
    """

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        self.image_token = "<image>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        # Get DeepSeek-OCR specific attributes from image_processor
        self.base_size = getattr(image_processor, 'base_size', 1024)
        self.image_size = getattr(image_processor, 'image_size', 768)
        self.patch_size = getattr(image_processor, 'patch_size', 16)
        self.downsample_ratio = getattr(image_processor, 'downsample_ratio', 4)
        self.add_image_token = True
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        **kwargs: Unpack[DeepseekOcrProcessorKwargs],
    ) -> BatchFeature:
        r"""
        Process images and text for DeepSeek-OCR model.

        Args:
            images: PIL images or image paths to process.
            text: Text prompts to tokenize.
            videos: Not supported for DeepSeek-OCR.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:
                - **input_ids** -- Token ids for the text input.
                - **attention_mask** -- Attention mask for text.
                - **images_ori** -- Original/global view images tensor.
                - **images_crop** -- Cropped/local view images tensor.
                - **images_spatial_crop** -- Spatial crop information.
                - **images_seq_mask** -- Mask indicating image token positions.
        """
        output_kwargs = self._merge_kwargs(
            DeepseekOcrProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        # Process images
        if self.add_image_token:
            add_image_token_id = [self.image_token_id]
        else:
            add_image_token_id = []
        text_splits = text.split(self.image_token)

        images_list, images_crop_list, images_seq_mask = [], [], []
        tokenized_str = []
        images_spatial_crop = []
        for text_sep, image in zip(text_splits, images):
            tokenized_sep = text_encode(self.tokenizer, text_sep, bos=False, eos=False)
            tokenized_str += tokenized_sep
            images_seq_mask += [False] * len(tokenized_sep)
            
            image_inputs = self.image_processor(images=image, add_image_token_id=add_image_token_id, **output_kwargs["images_kwargs"])
            images_list.extend(image_inputs["images_list"])
            images_crop_list.extend(image_inputs["images_crop_list"])
            images_spatial_crop.extend(image_inputs["images_spatial_crop"])
            images_seq_mask.extend(image_inputs["images_seq_mask"])
            tokenized_str += [self.image_token_id] * len(image_inputs["tokenized_image"])

        """process the last text split"""
        tokenized_sep = text_encode(self.tokenizer, text_splits[-1], bos=False, eos=False)
        tokenized_str += tokenized_sep
        images_seq_mask += [False] * len(tokenized_sep)

        # """add the bos tokens"""
        # bos_id = 0
        # tokenized_str = [bos_id] + tokenized_str 
        # images_seq_mask = [False] + images_seq_mask

        input_ids = torch.LongTensor(tokenized_str)
        images_seq_mask = torch.tensor(images_seq_mask, dtype=torch.bool)

        if len(images_list) == 0:
            images_ori = torch.zeros((1, 3, self.image_size, self.image_size))
            images_spatial_crop = torch.zeros((1, 2), dtype=torch.long)
            images_crop = torch.zeros((1, 3, self.base_size, self.base_size))

        else:
            images_ori = torch.stack(images_list, dim=0)
            images_spatial_crop = torch.tensor(images_spatial_crop, dtype=torch.long)
            if images_crop_list:
                images_crop = torch.stack(images_crop_list, dim=0)
            else:
                images_crop = torch.zeros((1, 3, self.base_size, self.base_size))
        
        return BatchFeature(data={
            "input_ids": input_ids.unsqueeze(0),
            "images_ori": images_ori,
            "images_crop": images_crop,
            "images_spatial_crop": images_spatial_crop,
            "images_seq_mask": images_seq_mask.unsqueeze(0),
        })

    def _get_num_multimodal_tokens(self, image_sizes=None, video_sizes=None, **kwargs):
        """Compute number of multimodal tokens needed."""
        vision_data = {}
        if image_sizes is not None:
            images_kwargs = DeepseekOcrProcessorKwargs._defaults.get("images_kwargs", {})
            images_kwargs.update(kwargs)

            num_image_patches = [
                self.image_processor.get_number_of_image_patches(*image_size, images_kwargs)
                for image_size in image_sizes
            ]
            vision_data.update({"num_image_patches": num_image_patches})

        return MultiModalData(**vision_data)

    def post_process_image_text_to_text(
        self, generated_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False, **kwargs
    ):
        """Post-process model output to decode text."""
        return self.tokenizer.batch_decode(
            generated_outputs,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    @property
    def model_input_names(self):
        """Return list of model input names."""
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names)) + ["images_seq_mask"]


__all__ = ["DeepseekOcrProcessor"]
