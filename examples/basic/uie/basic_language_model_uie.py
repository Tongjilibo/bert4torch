from bert4torch.pipelines import UIEPredictor
from pprint import pprint


if __name__ == '__main__':
    # 命名实体识别
    schema = ['时间', '选手', '赛事名称'] # Define the schema for entity extraction
    ie = UIEPredictor(model_path='E:/pretrain_ckpt/uie/uie_base_pytorch', schema=schema)
    pprint(ie("2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！"))

    schema = ['肿瘤的大小', '肿瘤的个数', '肝癌级别', '脉管内癌栓分级']
    ie.set_schema(schema)
    pprint(ie("（右肝肿瘤）肝细胞性肝癌（II-III级，梁索型和假腺管型），肿瘤包膜不完整，紧邻肝被膜，侵及周围肝组织，未见脉管内癌栓（MVI分级：M0级）及卫星子灶形成。（肿物1个，大小4.2×4.0×2.8cm）。"))

    # 关系抽取
    schema = {'竞赛名称': ['主办方', '承办方', '已举办次数']}
    ie.set_schema(schema) # Reset schema
    pprint(ie('2022语言与智能技术竞赛由中国中文信息学会和中国计算机学会联合主办，百度公司、中国中文信息学会评测工作委员会和中国计算机学会自然语言处理专委会承办，已连续举办4届，成为全球最热门的中文NLP赛事之一。'))

    # 事件抽取
    schema = {'地震触发词': ['地震强度', '时间', '震中位置', '震源深度']}
    ie.set_schema(schema) # Reset schema
    ie('中国地震台网正式测定：5月16日06时08分在云南临沧市凤庆县(北纬24.34度，东经99.98度)发生3.5级地震，震源深度10千米。')

    # 评论观点抽取
    schema = {'评价维度': ['观点词', '情感倾向[正向，负向]']}
    ie.set_schema(schema) # Reset schema
    pprint(ie("店面干净，很清静，服务员服务热情，性价比很高，发现收银台有排队"))

    # 情感倾向分类
    schema = '情感倾向[正向，负向]'
    ie.set_schema(schema)
    ie('这个产品用起来真的很流畅，我非常喜欢')