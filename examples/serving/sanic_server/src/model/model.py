from src.utils import loggers
import src.config.constants as constants
from src.utils.configs import Configuration
import traceback
from bert4torch.tokenizers import Tokenizer
from bert4torch.snippets import sequence_padding
import numpy as np

class BertModel():
    def __init__(self):
        rec_info_config = Configuration.configurations.get(constants.CONFIG_MODEL_KEY)
        self.model_path = rec_info_config.get(constants.CONFIG_MODEL_PATH)
        self.vocab_path = rec_info_config.get(constants.CONFIG_MODEL_VOCAB)
        self.mapping = {0: 'negative', 1: 'positive'}

    def load_model(self):
        try:
            import onnxruntime
            self.model = onnxruntime.InferenceSession(self.model_path)
            self.tokenizer = Tokenizer(self.vocab_path, do_lower_case=True)
        except Exception as ex:
            loggers.get_error_log().error("An exception occured while load model: {}".format(traceback.format_exc()))

    async def process(self, user_inputs):
        user_inputs = [user_inputs] if isinstance(user_inputs, str) else user_inputs
        input_ids, segment_ids = self.tokenizer.encode(user_inputs)
        input_ids = sequence_padding(input_ids).astype('int64')
        segment_ids = sequence_padding(segment_ids).astype('int64')

        # 模型推理结果
        ort_inputs = {self.model.get_inputs()[0].name: input_ids,
                    self.model.get_inputs()[1].name: segment_ids}
        ort_outs = self.model.run(None, ort_inputs)
        ort_outs = list(np.argmax(ort_outs[0], axis=1))
        return [{k:v} for k, v in zip(user_inputs, [self.mapping[i] for i in ort_outs])]