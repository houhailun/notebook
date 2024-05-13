#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   model_download.py
@Time    :   2024/01/19 11:09:12
@Author  :   Houhailun
@Version :   1.0
@Contact :   
@License :   
@Desc    :   None
'''

# 下载模型
# here put the import lib
from modelscope.hub.snapshot_download import snapshot_download

# model_dir = snapshot_download('tiansz/bert-base-chinese', cache_dir=r'D:\code\personal\models\bert-base-chinese')
model_dir = snapshot_download('sdfdsfe/bert-base-uncased', cache_dir=r'D:\code\personal\models\bert-base-uncased')
# modelscope会将模型和数据集下载到该环境变量指定的目录中
