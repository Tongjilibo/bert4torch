'''uie'''

import pytest
import torch
from bert4torch.pipelines import UIEPredictor
import os


@pytest.mark.parametrize("model_dir", ['E:/pretrain_ckpt/uie/uie_base_pytorch'])
@torch.inference_mode()
def test_uie(model_dir):
    # 情感倾向分类
    schema = '情感倾向[正向，负向]'
    ie = UIEPredictor(model_path=model_dir, schema=schema)
    ie.set_schema(schema)
    res = ie('这个产品用起来真的很流畅，我非常喜欢')
    print(res)
    assert res == [{'情感倾向[正向，负向]': [{'text': '正向', 'probability': 0.9990023970603943}]}]


if __name__ == '__main__':
    test_uie('E:/pretrain_ckpt/uie/uie_base_pytorch')