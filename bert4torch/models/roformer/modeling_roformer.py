from bert4torch.models.base import BertBase, register_model
from bert4torch.layers import LayerNorm
from bert4torch.snippets import delete_arguments
import re
from functools import partial


@register_model(name="roformer")
class RoFormer(BertBase):
    """旋转式位置编码的BERT模型；
    链接：https://kexue.fm/archives/8265
    """
    def load_variable(self, variable, ckpt_key, model_key, prefix='roformer'):
        return super().load_variable(variable, ckpt_key, model_key, prefix)
    
    def variable_mapping(self):
        mapping =  super().variable_mapping(prefix='roformer')
        del mapping['embeddings.position_embeddings.weight'] # 没有位置编码
        return mapping


@register_model(name="roformer_v2")
class RoFormerV2(RoFormer):
    """RoFormerV2；
    改动：去掉bias，简化Norm，优化初始化等。目前初始化暂时还用的bert的初始化，finetune不受影响
    """
    @delete_arguments('with_pool', 'with_nsp')
    def __init__(self, *args, **kwargs):
        super(RoFormerV2, self).__init__(*args, **kwargs)

        # LayerNorm没有weight
        for layer in self.modules():
            if isinstance(layer, LayerNorm) and hasattr(layer, 'weight'):
                del layer.weight
            
        if self.with_mlm:
            del self.mlmLayerNorm
            del self.mlmBias
            del self.mlmDense
            self.mlmDecoder.register_parameter('bias', None)

    def variable_mapping(self):
        mapping = super().variable_mapping()
        mapping_new = {}
        for k, v in mapping.items():
            if (not re.search('bias|layernorm', k.lower())) and (not re.search('bias|layernorm', v.lower())):
                mapping_new[k] = v
        return mapping_new

    def apply_final_layers(self, **model_kwargs):
        """根据剩余参数决定输出
        """
        # 获取最后一层隐藏层的输出
        encoded_layers = model_kwargs['encoded_layers']
        last_hidden_state = encoded_layers[-1]
        # 是否取最后一层输出
        if not self.output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        # 是否添加mlm
        if self.with_mlm:
            mlm_scores = self.mlmDecoder(last_hidden_state)
        else:
            mlm_scores = None
            
        # 是否取最后一层输出
        if not self.output_all_encoded_layers:
            return self.gen_outputs(locals(), last_hidden_state, mlm_scores)
        else:
            return self.gen_outputs(locals(), encoded_layers, mlm_scores)        
