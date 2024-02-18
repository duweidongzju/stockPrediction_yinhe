#!C:\pythonCode
# -*- coding: utf-8 -*-
# @Time : 2022/8/16 11:27
# @Author : duweidong
# @File : utils_duweidong.py
# @Software: PyCharm

"""
将一些常用的操作写成函数，方便以后调用

Typical usage example:

a = [[1, 2 ,3, 4, 5], [4, 'd', 3, 5, 6]]
a = read_txt(txt='files/t.txt')
write_txt('files/t.txt', a)
"""

import os
from typing import List, Dict, Tuple
import json
import pickle


def read_txt(txt: str, separator: str = None) -> List[List]:
    """
    the separator is blank, or tabs:'\t'

    Args:
        txt: path of txt file
        separator:

    Returns:

    """

    texture = []
    with open(txt, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            texture.append(line.strip().split(separator))

    return texture


def write_txt(txt: str, texture: List[List], separator: str = None) -> None:
    """
    the separator is blank, or tabs:'\t'

    Args:
        txt:
        texture:
        separator:

    Returns:

    """
    separator = '\t' if not separator else separator

    with open(txt, 'w', encoding='utf-8') as f:
        for line in texture:
            line = [str(item) for item in line]
            f.write(separator.join(line) + '\n')


def write_dict(json_path, dictionary):
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(dictionary, f, ensure_ascii=False)


def read_dict(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        dictionary = json.load(f)
    return dictionary


# def write_dict(json_path, dictionary):
#     with open(json_path, 'wb', encoding='utf-8') as f:
#         pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)
#
#
# def read_dict(json_path):
#     with open(json_path, 'rb', encoding='utf-8') as f:
#         dictionary = pickle.load(f)
#     return dictionary

