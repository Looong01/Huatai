# Huatai
华泰量化交易中的深度学习模型高性能推理部署


#### Introduction
为了简化、加速和优化深度学习模型在推理任务的部署，即，兼顾兼容性、易用性和时效性。我们选择了同一将模型转换为ONNX格式，并使用在ONNX Runtime推理框架进行了优化和部署。


#### Build & Run
一、模型转换  
For any system:  
环境配置:  
```
1. python -m pip install -U tensorflow onnxruntime tf2onnx
```
转换:
```
1. cd model
2. python -m tf2onnx.convert --saved-model 'fintech_model' --output 'fintech_model.onnx' --tag serve
```

二、推理  
For Linux:  
环境配置:  
```
1. sudo apt update
2. sudo apt upgrade
3. sudo apt install build-essential gcc g++ cmake make
4. wget "https://github.com/microsoft/onnxruntime/releases/download/v1.18.1/onnxruntime-linux-x64-1.18.1.tgz"
5. tar zxvf onnxruntime-linux-x64-1.18.1.tgz && rm -f onnxruntime-linux-x64-1.18.1.tgz
```
从源代码构建:  
```
1. mkdir build && cd build
2. cmake .. -DONNXRUNTIME_ROOTDIR=./onnxruntime-linux-x64-1.18.1.tgz && make
3. ./huatai ../model/fintech_model.onnx 16
```
测试预构建与生成的可执行文件:  
```
1. ./huatai model/fintech_model.onnx 16
```


#### 比赛要求

量化交易中的深度学习模型高性能推理部署 - 解码赛道（技术方向）
背景：

随着金融科技的快速发展，量化交易已成为金融市场的重要组成部分。量化交易依赖于先进的算法和模型，通过数据分析和预测来制定交易策略。近年来，深度学习模型因其和模式识别方面的优越性能，被广泛应用于量化交易中。然而，深度学习模型通常需要强大的计算资源和优化的部署策略，以确保在实时交易中的高效推理。



挑战：

在本赛题中，要求参赛选手设计和实现一个高性能的实时深度学习模型推理方案。该方案需要基于提供的深度学习模型，并能够在有限的计算资源下，实现高效、低延迟的推理性能。参赛选手需考虑如何优化模型的推理速度，同时确保模型的预测准确性,



课题内容：

根据提供的深度学习模型，基于C或C++等语言实现推理方案。须在确保推理结果正确的同时，尽可能地提升单样本下的推理速度。



成果要求：

基于C或C++等语言实现推理方案，可以采用现有开源引擎，但更推荐自研框架。测试和性能评价环境为Intel(R) Xeon(R) Platinum 8268 CPU、128GB内存。

成果形式要求：
1、推理代码和依赖库，并提供运行示例。

2、提供作品介绍：展示推理方案的设计思路、优势亮点、优化前后的性能提升等。形式及篇幅不限，视频/音频/文档/PPT等形式均可，表述清晰即可。



课题参考资料下载：
[【量化交易中的深度学习模型高性能推理部署】课题参考资料.zip](https://uploadfiles.nowcoder.com/files/20240530/328440_1717063678881/%E9%87%8F%E5%8C%96%E4%BA%A4%E6%98%93%E4%B8%AD%E7%9A%84%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A8%A1%E5%9E%8B%E9%AB%98%E6%80%A7%E8%83%BD%E6%8E%A8%E7%90%86%E9%83%A8%E7%BD%B2%E8%AF%BE%E9%A2%98%E5%8F%82%E8%80%83%E8%B5%84%E6%96%99.zip)
