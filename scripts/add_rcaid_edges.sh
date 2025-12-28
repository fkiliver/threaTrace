#!/bin/bash
# 批处理脚本：为多个数据集添加 RCAID 边并复制到指定位置

echo "开始处理所有数据集..."

# theia train
echo "处理 theia_train..."
python parse_darpatc_special.py \
    -i ta1-theia-e3-official-1r.json.txt \
    -o rcaid-add-edge/ta1-theia-e3-official-1r_with_rcaid.txt \
    -c ../graphchi-cpp-master/graph_data/darpatc/theia_train_rcaid.txt

# theia test
echo "处理 theia_test..."
python parse_darpatc_special.py \
    -i ta1-theia-e3-official-6r.json.8.txt \
    -o rcaid-add-edge/ta1-theia-e3-official-6r.8_with_rcaid.txt \
    -c ../graphchi-cpp-master/graph_data/darpatc/theia_test_rcaid.txt

# cadets train
echo "处理 cadets_train..."
python parse_darpatc_special.py \
    -i ta1-cadets-e3-official.json.1.txt \
    -o rcaid-add-edge/ta1-cadets-e3-official.1_with_rcaid.txt \
    -c ../graphchi-cpp-master/graph_data/darpatc/cadets_train_rcaid.txt

# cadets test
echo "处理 cadets_test..."
python parse_darpatc_special.py \
    -i ta1-cadets-e3-official-2.json.txt \
    -o rcaid-add-edge/ta1-cadets-e3-official-2_with_rcaid.txt \
    -c ../graphchi-cpp-master/graph_data/darpatc/cadets_test_rcaid.txt

# trace train
echo "处理 trace_train..."
python parse_darpatc_special.py \
    -i ta1-trace-e3-official-1.json.txt \
    -o rcaid-add-edge/ta1-trace-e3-official-1_with_rcaid.txt \
    -c ../graphchi-cpp-master/graph_data/darpatc/trace_train_rcaid.txt

# trace test
echo "处理 trace_test..."
python parse_darpatc_special.py \
    -i ta1-trace-e3-official-1.json.4.txt \
    -o rcaid-add-edge/ta1-trace-e3-official-1.4_with_rcaid.txt \
    -c ../graphchi-cpp-master/graph_data/darpatc/trace_test_rcaid.txt

echo "所有处理完成！"

