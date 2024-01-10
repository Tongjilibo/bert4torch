'''下载预训练模型并且转了pytorch格式
'''
import argparse
import collections
import json
import os
import pickle
import torch
import logging
import shutil
from tqdm import tqdm
import time

logger = logging.Logger('log')


def get_path_from_url(url, root_dir, check_exist=True, decompress=True):
    """ Download from given url to root_dir.
    if file or directory specified by url is exists under
    root_dir, return the path directly, otherwise download
    from url and decompress it, return the path.

    Args:
        url (str): download url
        root_dir (str): root dir for downloading, it should be
                        WEIGHTS_HOME or DATASET_HOME
        decompress (bool): decompress zip or tar file. Default is `True`

    Returns:
        str: a local path to save downloaded models & weights & datasets.
    """

    import os.path
    import os
    import tarfile
    import zipfile

    def is_url(path):
        """
        Whether path is URL.
        Args:
            path (string): URL string or not.
        """
        return path.startswith('http://') or path.startswith('https://')

    def _map_path(url, root_dir):
        # parse path after download under root_dir
        fname = os.path.split(url)[-1]
        fpath = fname
        return os.path.join(root_dir, fpath)

    def _get_download(url, fullname):
        import requests
        # using requests.get method
        fname = os.path.basename(fullname)
        try:
            req = requests.get(url, stream=True)
        except Exception as e:  # requests.exceptions.ConnectionError
            logger.info("Downloading {} from {} failed with exception {}".format(
                fname, url, str(e)))
            return False

        if req.status_code != 200:
            raise RuntimeError("Downloading from {} failed with code "
                               "{}!".format(url, req.status_code))

        # For protecting download interupted, download to
        # tmp_fullname firstly, move tmp_fullname to fullname
        # after download finished
        tmp_fullname = fullname + "_tmp"
        total_size = req.headers.get('content-length')
        with open(tmp_fullname, 'wb') as f:
            if total_size:
                with tqdm(total=(int(total_size) + 1023) // 1024, unit='KB') as pbar:
                    for chunk in req.iter_content(chunk_size=1024):
                        f.write(chunk)
                        pbar.update(1)
            else:
                for chunk in req.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        shutil.move(tmp_fullname, fullname)

        return fullname

    def _download(url, path):
        """
        Download from url, save to path.

        url (str): download url
        path (str): download to given path
        """

        if not os.path.exists(path):
            os.makedirs(path)

        fname = os.path.split(url)[-1]
        fullname = os.path.join(path, fname)
        retry_cnt = 0

        logger.info("Downloading {} from {}".format(fname, url))
        DOWNLOAD_RETRY_LIMIT = 3
        while not os.path.exists(fullname):
            if retry_cnt < DOWNLOAD_RETRY_LIMIT:
                retry_cnt += 1
            else:
                raise RuntimeError("Download from {} failed. "
                                   "Retry limit reached".format(url))

            if not _get_download(url, fullname):
                time.sleep(1)
                continue

        return fullname

    def _uncompress_file_zip(filepath):
        with zipfile.ZipFile(filepath, 'r') as files:
            file_list = files.namelist()

            file_dir = os.path.dirname(filepath)

            if _is_a_single_file(file_list):
                rootpath = file_list[0]
                uncompressed_path = os.path.join(file_dir, rootpath)
                files.extractall(file_dir)

            elif _is_a_single_dir(file_list):
                # `strip(os.sep)` to remove `os.sep` in the tail of path
                rootpath = os.path.splitext(file_list[0].strip(os.sep))[0].split(
                    os.sep)[-1]
                uncompressed_path = os.path.join(file_dir, rootpath)

                files.extractall(file_dir)
            else:
                rootpath = os.path.splitext(filepath)[0].split(os.sep)[-1]
                uncompressed_path = os.path.join(file_dir, rootpath)
                if not os.path.exists(uncompressed_path):
                    os.makedirs(uncompressed_path)
                files.extractall(os.path.join(file_dir, rootpath))

            return uncompressed_path

    def _is_a_single_file(file_list):
        if len(file_list) == 1 and file_list[0].find(os.sep) < 0:
            return True
        return False

    def _is_a_single_dir(file_list):
        new_file_list = []
        for file_path in file_list:
            if '/' in file_path:
                file_path = file_path.replace('/', os.sep)
            elif '\\' in file_path:
                file_path = file_path.replace('\\', os.sep)
            new_file_list.append(file_path)

        file_name = new_file_list[0].split(os.sep)[0]
        for i in range(1, len(new_file_list)):
            if file_name != new_file_list[i].split(os.sep)[0]:
                return False
        return True

    def _uncompress_file_tar(filepath, mode="r:*"):
        with tarfile.open(filepath, mode) as files:
            file_list = files.getnames()

            file_dir = os.path.dirname(filepath)

            if _is_a_single_file(file_list):
                rootpath = file_list[0]
                uncompressed_path = os.path.join(file_dir, rootpath)
                files.extractall(file_dir)
            elif _is_a_single_dir(file_list):
                rootpath = os.path.splitext(file_list[0].strip(os.sep))[0].split(
                    os.sep)[-1]
                uncompressed_path = os.path.join(file_dir, rootpath)
                files.extractall(file_dir)
            else:
                rootpath = os.path.splitext(filepath)[0].split(os.sep)[-1]
                uncompressed_path = os.path.join(file_dir, rootpath)
                if not os.path.exists(uncompressed_path):
                    os.makedirs(uncompressed_path)

                files.extractall(os.path.join(file_dir, rootpath))

            return uncompressed_path

    def _decompress(fname):
        """
        Decompress for zip and tar file
        """
        logger.info("Decompressing {}...".format(fname))

        # For protecting decompressing interupted,
        # decompress to fpath_tmp directory firstly, if decompress
        # successed, move decompress files to fpath and delete
        # fpath_tmp and remove download compress file.

        if tarfile.is_tarfile(fname):
            uncompressed_path = _uncompress_file_tar(fname)
        elif zipfile.is_zipfile(fname):
            uncompressed_path = _uncompress_file_zip(fname)
        else:
            raise TypeError("Unsupport compress file type {}".format(fname))

        return uncompressed_path

    assert is_url(url), "downloading from {} not a url".format(url)
    fullpath = _map_path(url, root_dir)
    if os.path.exists(fullpath) and check_exist:
        logger.info("Found {}".format(fullpath))
    else:
        fullpath = _download(url, root_dir)

    if decompress and (tarfile.is_tarfile(fullpath) or
                       zipfile.is_zipfile(fullpath)):
        fullpath = _decompress(fullpath)

    return fullpath


MODEL_MAP = {
    "uie-base": {
        "resource_file_urls": {
            "model_state.pdparams":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base_v0.1/model_state.pdparams",
            "model_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/model_config.json",
            "vocab_file":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
            "special_tokens_map":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/special_tokens_map.json",
            "tokenizer_config":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/tokenizer_config.json"
        }
    },
    "uie-medium": {
        "resource_file_urls": {
            "model_state.pdparams":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_medium_v1.0/model_state.pdparams",
            "model_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_medium/model_config.json",
            "vocab_file":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
            "special_tokens_map":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/special_tokens_map.json",
            "tokenizer_config":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/tokenizer_config.json",
        }
    },
    "uie-mini": {
        "resource_file_urls": {
            "model_state.pdparams":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_mini_v1.0/model_state.pdparams",
            "model_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_mini/model_config.json",
            "vocab_file":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
            "special_tokens_map":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/special_tokens_map.json",
            "tokenizer_config":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/tokenizer_config.json",
        }
    },
    "uie-micro": {
        "resource_file_urls": {
            "model_state.pdparams":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_micro_v1.0/model_state.pdparams",
            "model_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_micro/model_config.json",
            "vocab_file":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
            "special_tokens_map":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/special_tokens_map.json",
            "tokenizer_config":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/tokenizer_config.json",
        }
    },
    "uie-nano": {
        "resource_file_urls": {
            "model_state.pdparams":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_nano_v1.0/model_state.pdparams",
            "model_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_nano/model_config.json",
            "vocab_file":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
            "special_tokens_map":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/special_tokens_map.json",
            "tokenizer_config":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/tokenizer_config.json",
        }
    },
    "uie-medical-base": {
        "resource_file_urls": {
            "model_state.pdparams":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_medical_base_v0.1/model_state.pdparams",
            "model_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/model_config.json",
            "vocab_file":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
            "special_tokens_map":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/special_tokens_map.json",
            "tokenizer_config":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/tokenizer_config.json",
        }
    },
    "uie-tiny": {
        "resource_file_urls": {
            "model_state.pdparams":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_tiny_v0.1/model_state.pdparams",
            "model_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_tiny/model_config.json",
            "vocab_file":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_tiny/vocab.txt",
            "special_tokens_map":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_tiny/special_tokens_map.json",
            "tokenizer_config":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_tiny/tokenizer_config.json"
        }
    }
}


def build_params_map(attention_num=12):
    """
    build params map from paddle-paddle's ERNIE to transformer's BERT
    :return:
    """
    weight_map = collections.OrderedDict({
        'encoder.embeddings.word_embeddings.weight': "bert.embeddings.word_embeddings.weight",
        'encoder.embeddings.position_embeddings.weight': "bert.embeddings.position_embeddings.weight",
        'encoder.embeddings.token_type_embeddings.weight': "bert.embeddings.token_type_embeddings.weight",
        'encoder.embeddings.task_type_embeddings.weight': "embeddings.task_type_embeddings.weight",  # 这里没有前缀bert，直接映射到bert4torch结构
        'encoder.embeddings.layer_norm.weight': 'bert.embeddings.LayerNorm.weight',
        'encoder.embeddings.layer_norm.bias': 'bert.embeddings.LayerNorm.bias',
    })
    # add attention layers
    for i in range(attention_num):
        weight_map[f'encoder.encoder.layers.{i}.self_attn.q_proj.weight'] = f'bert.encoder.layer.{i}.attention.self.query.weight'
        weight_map[f'encoder.encoder.layers.{i}.self_attn.q_proj.bias'] = f'bert.encoder.layer.{i}.attention.self.query.bias'
        weight_map[f'encoder.encoder.layers.{i}.self_attn.k_proj.weight'] = f'bert.encoder.layer.{i}.attention.self.key.weight'
        weight_map[f'encoder.encoder.layers.{i}.self_attn.k_proj.bias'] = f'bert.encoder.layer.{i}.attention.self.key.bias'
        weight_map[f'encoder.encoder.layers.{i}.self_attn.v_proj.weight'] = f'bert.encoder.layer.{i}.attention.self.value.weight'
        weight_map[f'encoder.encoder.layers.{i}.self_attn.v_proj.bias'] = f'bert.encoder.layer.{i}.attention.self.value.bias'
        weight_map[f'encoder.encoder.layers.{i}.self_attn.out_proj.weight'] = f'bert.encoder.layer.{i}.attention.output.dense.weight'
        weight_map[f'encoder.encoder.layers.{i}.self_attn.out_proj.bias'] = f'bert.encoder.layer.{i}.attention.output.dense.bias'
        weight_map[f'encoder.encoder.layers.{i}.norm1.weight'] = f'bert.encoder.layer.{i}.attention.output.LayerNorm.weight'
        weight_map[f'encoder.encoder.layers.{i}.norm1.bias'] = f'bert.encoder.layer.{i}.attention.output.LayerNorm.bias'
        weight_map[f'encoder.encoder.layers.{i}.linear1.weight'] = f'bert.encoder.layer.{i}.intermediate.dense.weight'
        weight_map[f'encoder.encoder.layers.{i}.linear1.bias'] = f'bert.encoder.layer.{i}.intermediate.dense.bias'
        weight_map[f'encoder.encoder.layers.{i}.linear2.weight'] = f'bert.encoder.layer.{i}.output.dense.weight'
        weight_map[f'encoder.encoder.layers.{i}.linear2.bias'] = f'bert.encoder.layer.{i}.output.dense.bias'
        weight_map[f'encoder.encoder.layers.{i}.norm2.weight'] = f'bert.encoder.layer.{i}.output.LayerNorm.weight'
        weight_map[f'encoder.encoder.layers.{i}.norm2.bias'] = f'bert.encoder.layer.{i}.output.LayerNorm.bias'
    # add pooler
    weight_map.update(
        {
            'encoder.pooler.dense.weight': 'bert.pooler.dense.weight',
            'encoder.pooler.dense.bias': 'bert.pooler.dense.bias',
            'linear_start.weight': 'linear_start.weight',
            'linear_start.bias': 'linear_start.bias',
            'linear_end.weight': 'linear_end.weight',
            'linear_end.bias': 'linear_end.bias',
        }
    )
    return weight_map


def extract_and_convert(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger.info('=' * 20 + 'save config file' + '=' * 20)
    config = json.load(open(os.path.join(input_dir, 'model_config.json'), 'rt', encoding='utf-8'))
    config = config['init_args'][0]
    config["architectures"] = ["UIE"]
    config['layer_norm_eps'] = 1e-12
    del config['init_class']
    if 'sent_type_vocab_size' in config:
        config['type_vocab_size'] = config['sent_type_vocab_size']
    config['intermediate_size'] = 4 * config['hidden_size']
    json.dump(config, open(os.path.join(output_dir, 'config.json'),
              'wt', encoding='utf-8'), indent=4)
    logger.info('=' * 20 + 'save vocab file' + '=' * 20)
    with open(os.path.join(input_dir, 'vocab.txt'), 'rt', encoding='utf-8') as f:
        words = f.read().splitlines()
    words_set = set()
    words_duplicate_indices = []
    for i in range(len(words)-1, -1, -1):
        word = words[i]
        if word in words_set:
            words_duplicate_indices.append(i)
        words_set.add(word)
    for i, idx in enumerate(words_duplicate_indices):
        words[idx] = chr(0x1F6A9+i)  # Change duplicated word to 🚩 LOL
    with open(os.path.join(output_dir, 'vocab.txt'), 'wt', encoding='utf-8') as f:
        for word in words:
            f.write(word+'\n')
    special_tokens_map = {
        "unk_token": "[UNK]",
        "sep_token": "[SEP]",
        "pad_token": "[PAD]",
        "cls_token": "[CLS]",
        "mask_token": "[MASK]"
    }
    json.dump(special_tokens_map, open(os.path.join(output_dir, 'special_tokens_map.json'),
              'wt', encoding='utf-8'))
    tokenizer_config = {
        "do_lower_case": True,
        "unk_token": "[UNK]",
        "sep_token": "[SEP]",
        "pad_token": "[PAD]",
        "cls_token": "[CLS]",
        "mask_token": "[MASK]",
        "tokenizer_class": "BertTokenizer"
    }
    json.dump(tokenizer_config, open(os.path.join(output_dir, 'tokenizer_config.json'),
              'wt', encoding='utf-8'))
    logger.info('=' * 20 + 'extract weights' + '=' * 20)
    state_dict = collections.OrderedDict()
    weight_map = build_params_map(attention_num=config['num_hidden_layers'])
    paddle_paddle_params = pickle.load(
        open(os.path.join(input_dir, 'model_state.pdparams'), 'rb'))
    del paddle_paddle_params['StructuredToParameterName@@']
    for weight_name, weight_value in paddle_paddle_params.items():
        if 'weight' in weight_name:
            if 'encoder.encoder' in weight_name or 'pooler' in weight_name or 'linear' in weight_name:
                weight_value = weight_value.transpose()
        # Fix: embedding error
        if 'word_embeddings.weight' in weight_name:
            weight_value[0, :] = 0
        if weight_name not in weight_map:
            logger.info(f"{'='*20} [SKIP] {weight_name} {'='*20}")
            continue
        state_dict[weight_map[weight_name]] = torch.FloatTensor(weight_value)
        logger.info(f"{weight_name} -> {weight_map[weight_name]} {weight_value.shape}")
    torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))


def check_model(input_model):
    if not os.path.exists(input_model):
        if input_model not in MODEL_MAP:
            raise ValueError('input_model not exists!')

        resource_file_urls = MODEL_MAP[input_model]['resource_file_urls']
        logger.info("Downloading resource files...")

        for key, val in resource_file_urls.items():
            file_path = os.path.join(input_model, key)
            if not os.path.exists(file_path):
                get_path_from_url(val, input_model)


def do_main():
    check_model(args.input_model)
    extract_and_convert(args.input_model, args.output_model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_model", default="uie-base", type=str,
                        help="Directory of input paddle model.\n Will auto download model [uie-base/uie-tiny]")
    parser.add_argument("-o", "--output_model", default="uie_base_pytorch", type=str,
                        help="Directory of output pytorch model")
    args = parser.parse_args()

    do_main()
