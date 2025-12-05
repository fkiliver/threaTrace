#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""快速测试 UnitMerge 脚本"""

import sys
import os.path as osp

# 添加当前目录到路径
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from create_unitmerge_data import create_unitmerge_data

# 测试小文件
test_file = '../graphchi-cpp-master/graph_data/darpatc/trace_test.txt'
output_file = '../graphchi-cpp-master/graph_data/darpatc/trace_test_unitmerge_test.txt'

if osp.exists(test_file):
    print("开始测试...")
    create_unitmerge_data(test_file, output_file, remove_isolated=True, match_by_type=False)
    print("\n测试完成！")
else:
    print(f"测试文件不存在: {test_file}")

