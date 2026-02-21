import torch
from torch import nn
from abc import ABC
from typing import List, Optional, Tuple, Union, Dict
from torch4keras.snippets import safe_import
from PIL import Image, ImageOps
import math
with safe_import():
    from torchvision import transforms

def normalize_transform(mean, std):
    if mean is None and std is None:
        transform = None
    elif mean is None and std is not None:
        mean = [0.] * len(std)
        transform = transforms.Normalize(mean=mean, std=std)
    elif mean is not None and std is None:
        std = [1.] * len(mean)
        transform = transforms.Normalize(mean=mean, std=std)
    else:
        transform = transforms.Normalize(mean=mean, std=std)

    return transform


class BaseTransform(ABC):

    def set_rng(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        pass

    @property
    def default_shape(self):
        raise NotImplementedError


class BasicImageTransform(BaseTransform):
    def __init__(
        self, 
        mean: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
        std: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
        normalize: bool = True
    ):

        self.mean = mean
        self.std = std
    
        transform_pipelines = [
            transforms.ToTensor()
        ]

        normalize = normalize_transform(mean, std) if normalize else nn.Identity()
        if normalize is not None:
            transform_pipelines.append(normalize)

        self.transform = transforms.Compose(transform_pipelines)
    
    def __call__(self, x):
        x = self.transform(x)
        return x


def load_image(image_path):

    try:
        image = Image.open(image_path)
        
        corrected_image = ImageOps.exif_transpose(image)
        
        return corrected_image
        
    except Exception as e:
        print(f"error: {e}")
        try:
            return Image.open(image_path)
        except:
            return None
        

def load_pil_images(conversations: List[Dict[str, str]]) -> List[Image.Image]:
    """

    Args:
        conversations (List[Dict[str, str]]): the conversations with a list of messages. An example is :
            [
                {
                    "role": "User",
                    "content": "<image_placeholder>\nExtract all information from this image and convert them into markdown format.",
                    "images": ["./examples/table_datasets.png"]
                },
                {"role": "Assistant", "content": ""},
            ]

    Returns:
        pil_images (List[PIL.Image.Image]): the list of PIL images.

    """

    pil_images = []

    for message in conversations:
        if "images" not in message:
            continue

        for image_path in message["images"]:
            # print('----------------')
            # print(image_path)
            # print('----------------')
            # exit()
            
            # pil_img = Image.open(image_path)
            pil_img = load_image(image_path)
            pil_img = pil_img.convert("RGB")
            pil_images.append(pil_img)

    return pil_images

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    # print(f'width: {width}, height: {height}, best_ratio: {best_ratio}')
    return best_ratio


def dynamic_preprocess(image, min_num=2, max_num=6, image_size=768, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    # print(target_ratios)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # print(target_aspect_ratio)
    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images, target_aspect_ratio

def text_encode(tokenizer, text: str, bos: bool = True, eos: bool = False):
    t = tokenizer.encode(text, add_special_tokens=False)
    bos_id = 0
    eos_id = 1
    if bos:
        t = [bos_id] + t
    if eos:
        t = t + [eos_id]

    return t

def process_vision_info(tokenizer, prompt='', image_file='', base_size=1024, 
                        image_size=768, crop_mode=True, crop_thread=768, 
                        add_image_token_id=False, dynamic_preprocess_max_num=6):
    if not prompt:
        assert False, f'prompt is none!'
        
    conversation = [
        {
            "role": "<|User|>",
            "content": f'{prompt}',
        },
        {"role": "<|Assistant|>", "content": ""},
    ]
    if image_file:
        conversation[0]['images'] = [f'{image_file}']
    
    patch_size = 16
    downsample_ratio = 4
    images = load_pil_images(conversation)

    valid_img_tokens = 0
    ratio = 1

    image_draw = images[0].copy()

    w,h = image_draw.size
    ratio = 1 - ((max(w, h) - min(w, h)) / (max(w, h)))


    image_transform=BasicImageTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True)
    images_seq_mask = []

    image_token = '<image>'
    image_token_id = 128815
    if add_image_token_id:
        add_image_token_id = [image_token_id]
    else:
        add_image_token_id = []
    text_splits = prompt.split(image_token)

    images_list, images_crop_list, images_seq_mask = [], [], []
    tokenized_str = []
    images_spatial_crop = []
    for text_sep, image in zip(text_splits, images):

        tokenized_sep = text_encode(tokenizer, text_sep, bos=False, eos=False)
        tokenized_str += tokenized_sep
        images_seq_mask += [False] * len(tokenized_sep)

        if crop_mode:

            if image.size[0] <= crop_thread and image.size[1] <= crop_thread:
                crop_ratio = [1, 1]

            else:
                if crop_mode:
                    # best_width, best_height = select_best_resolution(image.size, self.candidate_resolutions)
                    images_crop_raw, crop_ratio = dynamic_preprocess(image, max_num=dynamic_preprocess_max_num, image_size=image_size)
                else:
                    # best_width, best_height = self.image_size, self.image_size
                    crop_ratio = [1, 1]
            
            """process the global view"""
            # image = image.resize((base_size, base_size))
            global_view = ImageOps.pad(image, (base_size, base_size),
                                    color=tuple(int(x * 255) for x in image_transform.mean))
            
            if base_size == 1024:
                valid_img_tokens += int(256 * ratio)
            elif base_size == 1280:
                valid_img_tokens += int(400 * ratio)
            # elif base_size == 640:
            #     valid_img_tokens += int(100 * ratio)
            
            images_list.append(image_transform(global_view).to(torch.bfloat16))

            # global_view_tensor = image_transform(global_view).to(torch.bfloat16)

            width_crop_num, height_crop_num = crop_ratio

            images_spatial_crop.append([width_crop_num, height_crop_num])
            
            
            if width_crop_num > 1 or height_crop_num > 1:
                """process the local views"""
                
                for i in range(len(images_crop_raw)):
                    images_crop_list.append(image_transform(images_crop_raw[i]).to(torch.bfloat16))
            
            if image_size == crop_thread:
                valid_img_tokens += len(images_crop_list) * 144

            num_queries = math.ceil((image_size // patch_size) / downsample_ratio)
            num_queries_base = math.ceil((base_size // patch_size) / downsample_ratio)

            """add image tokens"""
            tokenized_image = ([image_token_id] * num_queries_base + add_image_token_id) * num_queries_base
            tokenized_image += [image_token_id]
            if width_crop_num > 1 or height_crop_num > 1:
                tokenized_image += ([image_token_id] * (num_queries * width_crop_num) + add_image_token_id) * (
                            num_queries * height_crop_num)
            tokenized_str += tokenized_image
            images_seq_mask += [True] * len(tokenized_image)
            # num_image_tokens.append(len(tokenized_image))

        else:
            """process the global view"""
            if image_size <= crop_thread:
                print('directly resize')
                image = image.resize((image_size, image_size))
            # else:
            global_view = ImageOps.pad(image, (image_size, image_size),
                                    color=tuple(int(x * 255) for x in image_transform.mean))
            images_list.append(image_transform(global_view).to(torch.bfloat16))

            if base_size == 1024:
                valid_img_tokens += int(256 * ratio)
            elif base_size == 1280:
                valid_img_tokens += int(400 * ratio)
            elif base_size == 640:
                valid_img_tokens += int(100 * 1)
            elif base_size == 512:
                valid_img_tokens += int(64 * 1)
            elif base_size == 768:
                valid_img_tokens += int(144 * 1)

            width_crop_num, height_crop_num = 1, 1

            images_spatial_crop.append([width_crop_num, height_crop_num])


            """add image tokens"""
            num_queries = math.ceil((image_size // patch_size) / downsample_ratio)

            tokenized_image = ([image_token_id] * num_queries + add_image_token_id) * num_queries
            tokenized_image += [image_token_id]
            # tokenized_image += ([self.image_token_id] * (num_queries * width_crop_num) + [self.image_token_id]) * (
            #             num_queries * height_crop_num)
            tokenized_str += tokenized_image
            images_seq_mask += [True] * len(tokenized_image)
            # num_image_tokens.append(len(tokenized_image))
    

    """process the last text split"""
    tokenized_sep = text_encode(tokenizer, text_splits[-1], bos=False, eos=False)
    tokenized_str += tokenized_sep
    images_seq_mask += [False] * len(tokenized_sep)

    """add the bos tokens"""
    bos_id = 0
    tokenized_str = [bos_id] + tokenized_str 
    images_seq_mask = [False] + images_seq_mask

    input_ids = torch.LongTensor(tokenized_str)
    images_seq_mask = torch.tensor(images_seq_mask, dtype=torch.bool)

    if len(images_list) == 0:
        images_ori = torch.zeros((1, 3, image_size, image_size))
        images_spatial_crop = torch.zeros((1, 2), dtype=torch.long)
        images_crop = torch.zeros((1, 3, base_size, base_size))

    else:
        images_ori = torch.stack(images_list, dim=0)
        images_spatial_crop = torch.tensor(images_spatial_crop, dtype=torch.long)
        if images_crop_list:
            images_crop = torch.stack(images_crop_list, dim=0)
        else:
            images_crop = torch.zeros((1, 3, base_size, base_size))
    
    outputs = {
        'input_ids':input_ids.unsqueeze(0), 
        'images_ori': images_ori, 
        'images_crop': images_crop, 
        'images_spatial_crop': images_spatial_crop, 
        'images_seq_mask': images_seq_mask.unsqueeze(0)
        }
    return outputs