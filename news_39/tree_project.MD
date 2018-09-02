# 目录结果

```
├── crawler                                                         爬虫
│   ├── __init__.py
│   ├── clean_data.py
│   ├── crawler_index.py
│   ├── get_content.py
│   ├── get_page_url.py
├── data                                                            数据
│   ├── all_data
│   ├── ask_data
│   ├── clean_data
│   ├── index
│   │   └── index.json
│   ├── page_url
│   ├── train_data
│   ├── train_data_2
│   │   ├── train_data.csv
│   └── train_data_3
├── data_base                                                       知识图谱
│   ├── __init__.py
│   ├── data                                                        知识图谱数据
│   │   ├── __init__.py
│   ├── headers
│   │   ├── department_header.csv
│   │   ├── department_label_header.csv
│   │   └── label_header.csv
│   ├── import.sh                                                   知识图谱导入脚本
│   └── network.py                                                  构建知识图谱节点和关系的脚本
├── data_parse
│   ├── 111.py
│   └── __init__.py
├── lstm_rnn_classfy                                                文本分类
│   ├── __init__.py
│   ├── checkpoints
│   │   └── textrnn
│   ├── cnn_model_aa.py                                             cnn模型
│   ├── cnn_model_bb.py
│   ├── lstm_aa.py
│   ├── lstm_bb.py
│   ├── rnn_model_aa.py                                             rnn模型
│   ├── run_cnn_aa.py
│   ├── run_rnn_aa.py
│   └── tensorboard
│       └── textrnn
├── ner_data_prepare                                                实体抽取打标签
│   ├── __init__.py
│   └── tagging.py
├── ner_model                                                       实体抽取模型
│   ├── __init__.py
│   ├── data_utils.py
│   ├── ner_model
│   │   ├── saved_model.pb
│   │   └── variables
│   │       ├── variables.data-00000-of-00001
│   │       └── variables.index
│   ├── predict.py
│   ├── rnn_cnn_crf.py
│   └── train_model.py
├── service                                                         api服务
│   ├── __init__.py
│   ├── neo4j_utils.py
│   ├── ner_model_utils.py
│   └── server.py
└── test
    ├── __init__.py
    └── test_service
        ├── __init__.py
        └── test_server.py                                          api服务访问demo
```