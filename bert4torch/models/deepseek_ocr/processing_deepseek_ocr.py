from ...processor.feature_extraction_utils import BatchFeature
from ...processor.image_utils import ImageInput
from ...processor.processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack, register_processor
from ...tokenizers.tokenization_utils_base import PreTokenizedInput, TextInput
from ...processor.video_utils import VideoInput
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
        self.image_token = "<|image_pad|>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
        self.video_token = "<|video_pad|>" if not hasattr(tokenizer, "video_token") else tokenizer.video_token
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        self.video_token_id = (
            tokenizer.video_token_id
            if getattr(tokenizer, "video_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.video_token)
        )
        # Get DeepSeek-OCR specific attributes from image_processor
        self.patch_size = getattr(image_processor, 'patch_size', 16)
        self.downsample_ratio = getattr(image_processor, 'downsample_ratio', 4)
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        videos: VideoInput | None = None,
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
        image_inputs = {}
        images_seq_mask = []
        tokenized_results = []

        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs.get("images_spatial_crop")

            if not isinstance(text, list):
                text = [text]
            text = text.copy()

            # Calculate number of queries for image tokens
            num_queries = math.ceil((self.image_processor.image_size // self.patch_size) / self.downsample_ratio)
            num_queries_base = math.ceil((self.image_processor.base_size // self.patch_size) / self.downsample_ratio)

            image_token_id = self.image_processor.image_token_id if hasattr(self.image_processor, 'image_token_id') else self.image_token_id

            # Build text with image placeholders
            for i, single_text in enumerate(text):
                if self.image_token in single_text:
                    parts = single_text.split(self.image_token)
                    tokenized_parts = []
                    mask_parts = []

                    for j, part in enumerate(parts):
                        if part:
                            tokenized_part = text_encode(self.tokenizer, part, bos=False, eos=False)
                            tokenized_parts.extend(tokenized_part)
                            mask_parts.extend([False] * len(tokenized_part))

                        if j < len(parts) - 1:
                            # Add image token placeholder
                            width_crop_num, height_crop_num = image_grid_thw[i].tolist() if i < len(image_grid_thw) else [1, 1]

                            if width_crop_num > 1 or height_crop_num > 1:
                                tokenized_image = ([image_token_id] * num_queries_base + [image_token_id]) * num_queries_base
                                tokenized_image += ([image_token_id] * (num_queries * width_crop_num) + [image_token_id]) * (
                                    num_queries * height_crop_num)
                            else:
                                tokenized_image = ([image_token_id] * num_queries + [image_token_id]) * num_queries
                                tokenized_image += [image_token_id]

                            tokenized_parts.extend(tokenized_image)
                            mask_parts.extend([True] * len(tokenized_image))

                    # Add last text part with bos token
                    tokenized_str = [0] + tokenized_parts + text_encode(self.tokenizer, parts[-1], bos=False, eos=False) if parts[-1] else [0] + tokenized_parts
                    mask_str = [False] + mask_parts + [False] * len(text_encode(self.tokenizer, parts[-1], bos=False, eos=False)) if parts[-1] else [False] + mask_parts

                    tokenized_results.append(tokenized_str)
                    images_seq_mask.append(mask_str)
                else:
                    # No image token in text
                    tokenized_str = text_encode(self.tokenizer, single_text, bos=True, eos=False)
                    mask_str = [False] * len(tokenized_str)
                    tokenized_results.append(tokenized_str)
                    images_seq_mask.append(mask_str)

            # Pad tokenized results to same length
            max_len = max(len(t) for t in tokenized_results)
            padded_input_ids = []
            padded_mask = []
            for t, m in zip(tokenized_results, images_seq_mask):
                pad_len = max_len - len(t)
                padded_input_ids.append(t + [self.tokenizer.pad_token_id] * pad_len)
                padded_mask.append(m + [False] * pad_len)

            input_ids = torch.LongTensor(padded_input_ids)
            images_seq_mask = torch.tensor(padded_mask, dtype=torch.bool)
        else:
            # No images, just tokenize text
            if not isinstance(text, list):
                text = [text]
            text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
            return BatchFeature(data={**text_inputs}, tensor_type=output_kwargs["text_kwargs"].get("return_tensors"))

        return BatchFeature(data={
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
            "images_ori": image_inputs["images_ori"],
            "images_crop": image_inputs["images_crop"],
            "images_spatial_crop": image_inputs["images_spatial_crop"],
            "images_seq_mask": images_seq_mask,
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
