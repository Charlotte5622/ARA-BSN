'''
模型注册和管理模块

功能：
1. 提供模型注册机制
2. 维护可用模型的字典
3. 自动导入所有模型文件
'''

import os
from importlib import import_module

# 存储所有已注册的模型类
model_class_dict = {}

def regist_model(model_class):
    '''
    模型注册装饰器
    
    功能：
    - 将模型类注册到全局字典中
    - 确保模型名称唯一
    
    参数：
    - model_class: 要注册的模型类
    
    返回：
    - 被装饰的模型类
    '''
    model_name = model_class.__name__.lower()
    assert not model_name in model_class_dict, 'there is already registered model: %s in model_class_dict.' % model_name
    model_class_dict[model_name] = model_class
    
    # Special aliases for ARA-BSN variants
    if model_class.__name__ == 'ARABSN':
        model_class_dict['ara-bsn'] = model_class
        # Backward compatibility: register as APBSN for loading old models
        model_class_dict['apbsn'] = model_class

    return model_class

def get_model_class(model_name:str):
    '''
    获取已注册的模型类
    
    参数：
    - model_name: 模型名称（不区分大小写）
    
    返回：
    - 对应的模型类
    '''
    model_name = model_name.lower()
    return model_class_dict[model_name]

# 自动导入model文件夹下的所有Python文件
for module in os.listdir(os.path.dirname(__file__)):
    if module == '__init__.py' or module[-3:] != '.py':
        continue
    import_module('src.model.{}'.format(module[:-3]))
del module