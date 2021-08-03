# bytecamp-NER

## 简介
为了帮用户更准确地找到喜欢的房子，房产APP一般都提供找房的服务，但是用户的搜索需求总是很明确，比如搜索“知春东里”，意味着必须返回这个小区的房源，再比如用户搜索“中关村一小”，意味着用户需要寻找中关村一小的学区房；所以我们第一步需要去识别用户的意图，才能更好地帮助用户寻找到合适的房子，从而产生交易带来商业价值。本项目会利用现有用户的搜索日志、房源信息的数据，利用NLP技术，预估出用户的搜索意图。

本项目通过人工标注+正则提取+label匹配的方法从未标注的大量Query信息中提取出词典，并生成训练使用的标记数据。模型选择Bert+CRF的组合，训练后可对用户的Query进行分词和标注，分辨出用户想要的房源的各项信息。


## data

### 数据说明

#### byte_camp_house_info.txt

房源信息数据，按照\t 分割，按顺序依次是如下信息
| Info信息 |
| ---- |
| 房源标题 |
| 房源名称 |
| 房源单价（单位：分）|
| 所在楼层|
| 总楼层|
| 楼层位置，1高楼层；2中楼层，3低楼层，4顶层，5底层，6地下|
| 朝向，1’东’， 2’西’，3’南’ ，4 ’北’，5’东南’， 6 ’西南’， 7’东北’， 8’西北’ ，9’南北’|
| 户型，整体房间数|
| 户型，整体客厅数|
| 户型，整体卫生间数|
| 户型，整体厨房数|
| 装修程度，1精装修；2简装；3毛坯；99其他|
| 城市名称|
| 行政区名称|
| 商圈名称|
| 小区名称|
| 小区别名（列表）|
| 开发商名称|
样例数据
1.仙人阁附近湘中北苑精装房，带全套家具家电，钻石楼层！可按揭        仙人阁附近湘中北苑精装房，带全套家具家电，钻石楼层！可按揭        48800000        95.0        2        6        3        9        2        2        1        1        2        娄底        娄星        娄星        湘中北苑        []        \N


#### byte_camp_query.txt

真实用户query数据，按照\t 分割，按顺序依次是如下信息
| Query信息 |
| ---- |
| 搜索文本 |
| 当前城市 |
| 搜索次数 |
样例数据
1.南昌西站付近精装修的二手房          南昌          1


#### 输出
分词之后每个term的类型

需要识别的类型包括：
| 输出信息 |
| ---- |
| 行政区 |
| 商圈 |
| 小区 |
| 总价 |
| 面积 |
| 楼层 |
| 装修 |
| 开发商 |
| 房源属性 |
样例输出
海淀牡丹园金尚嘉园二居室南北
-> 海淀[行政区]牡丹园[商圈]金尚嘉园[小区]二居室[户型]南北[朝向]

长安区凤栖原地铁口附近
-> 长安区[行政区]凤栖原[地铁站]地铁口[标签]附近[无效]


### dict

通过data_process/build_dict.py建立词典，读入data/byte_camp_house_info.txt中的关于房源知识库中结构化数据，经过一些数据清洗后。生成的词典存放于/data/dict/目录下，包含data/city.txt, data/district.txt, data/business.txt, data/house.txt, data/developer.txt等文件

* data/district.txt: 地区的词典
* data/business.txt: 商圈的词典
* data/house.txt: 小区名/别名
* data/developer.txt: 开发商名字

#### 人工提取的词典

* data/house_attr.txt: 包含一些房源属性，比如：别墅，复式
* data/tag.txt: NER时使用的标签，比如：O，B-district，I-district...

### 数据处理detail

1. 数据清洗：大写转小写，全角转半角，去除query中的空格tab
2. 词典抽取：对于不同的词典，设置不同的阈值，将长度小于阈值的key删除
3. 房源属性词典：对所有query分词，统计每个word的频率，在高频词词中人工挑选房源属性

### 数据标记方法

通过data_process/label_data.py对data/byte_camp_query.txt的query进行机器标注。
使用了BertTokenizer进行分词。
Bert模型存放于prev_trained_model/chinese_wwm_ext_pytorch文件中。
通过按照词典从长到短贪心匹配后，将产生的标记数据输出到文件data/labeles_data.txt文件中


#### 需要正则匹配的tag

* 总价，`"(朝|向|坐|朝向|面向|方向|面朝)(北|东北|东|东南|南|西南|西|西北)+(方向|方)?"`
* 户型，`"(([1-9(一|二|三|四|五|六|七|八|九|十)]|独)(室|房))", "(([1-9(一|二|三|四|五|六|七|八|九|十)]|独)卫)", "(([1-9(一|二|三|四|五|六|七|八|九|十)]|独)厅)", "(([1-9(一|二|三|四|五|六|七|八|九|十)]|独)厨)"`
* 楼层，`"(大约|大概)?([1-9]+|(一|二|三|四|五|六|七|八|九|十|千|百)+)万([1-9]+|(一|二|三|四|五|六|七|八|九|十|千|百)+)?(块|元|元钱|块钱)?(左右)?(每平米|每平方米|每平)?(每月|每天|每日|每年)?(以上|以下)?"`
* 朝向，`"(地下)?([1-9]+|(一|二|三|四|五|六|七|八|九|十)+|顶|底)(楼|层)(以上|以下)?"`
* 面积，`"(大约|大概)?([1-9]+|(一|二|三|四|五|六|七|八|九|十)+(百|十)([1-9]+|(一|二|三|四|五|六|七|八|九|十)十)?)(平米|平方|平方米|平)(以上|以下)?"`


## data_postprocess

* build_pred_dict.py: 构造预测之后新产生的词典


## prev_trained_model

* 现成的预训练Bert模型[chinese_wwm_ext_pytorch](https://github.com/ymcui/Chinese-BERT-wwm)
  

## models

使用了Bert+CRF模型
* bert_for_ner.py: 结合了Bert+CRF的模型
  

### 参数设置

'--save_log', action='store_true'
'--checkpoint', type=str, default=None
'--learning_rate', type=float, default=0.001
'--step_size', type=int, default=1000
'--decay_gamma', type=float, default=0.5
'--n_epoch', type=int, default=10
'-b', '--batch_size', type=int, default=64
'--val_batch_size', type=int, default=64
'--print_every', type=int, default=50


### layers

* crf.py: 现成的CRF模型
  

## utils

一些小工具

* utils/clear_text.py: 对文本进行清洗
* utils/tag_dict.py: 用来将标签和id进行转化
* utils/score.py: 计算precision，recall和f1分数
* utils/load_data.py: 加载BIO标记的数据


## ckpt

保存模型的文件


## 其他文件

* config.py: 训练中的参数
* dataloader.py: dataloader对象
* eval.py: 利用带标记的query评估模型
* model_predict.py: 对输入的query文件进行预测
* regular_matching eval.py: 在取样标记数据上评估各项分数
* solver.py: 一个训练/验证模型的类
* test/py: 对输入的单句query进行预测
* train.py: 初始化config，dataloader，solver，进行训练
