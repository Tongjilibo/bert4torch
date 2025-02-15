from bert4torch.pipelines import FillMask
import pytest

@pytest.mark.parametrize("model_dir", ['E:/data/pretrain_ckpt/google-bert/bert-base-chinese'])
def test_fill_mask(model_dir):
    model = FillMask(model_dir)
    res = model.predict(["今天[MASK]气不错，[MASK]情很好", '[MASK]学技术是第一生产力'])
    assert res[0]['pred_token'] == ['天', '心']
    assert res[1]['pred_token'] == ['科']
    return res


if __name__ == '__main__':
    from pprint import pprint
    res = test_fill_mask('E:/data/pretrain_ckpt/google-bert/bert-base-chinese')
    pprint(res)