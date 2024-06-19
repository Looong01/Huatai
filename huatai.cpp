#include <iostream>
#include <vector>
#include <random>
#include <onnxruntime_cxx_api.h>

// 生成一个随机的浮点数矩阵
std::vector<float> generateRandomMatrix(size_t rows, size_t cols) {
    std::vector<float> matrix(rows * cols); // 创建一个大小为rows*cols的向量
    std::random_device rd;  // 随机设备，用于生成随机种子
    std::mt19937 gen(rd()); // 以随机设备rd生成的种子初始化梅森旋转发生器
    std::uniform_real_distribution<> dis(0.0, 1.0);  // 创建一个均匀分布的实数分布器

    for (size_t i = 0; i < rows * cols; ++i) {
        matrix[i] = static_cast<float>(dis(gen)); // 生成随机浮点数并存入矩阵
    }
    return matrix;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {  // 如果参数数量不等于3
        std::cerr << "Usage: " << argv[0] << " <model_path> <batch_size>" << std::endl;
        return 1;
    }

    const char* modelPath = argv[1];
    int batchSize = std::stoi(argv[2]);
    size_t inputDim = 100;

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ModelInference");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    Ort::Session session(env, modelPath, session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    // 生成随机输入
    std::vector<float> input_1 = generateRandomMatrix(batchSize, inputDim);
    std::vector<float> input_2 = generateRandomMatrix(batchSize, inputDim);

    // 创建输入张量
    std::vector<int64_t> inputShape = {batchSize, static_cast<int64_t>(inputDim)};
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value inputTensor_1 = Ort::Value::CreateTensor<float>(memoryInfo, input_1.data(), input_1.size(), inputShape.data(), inputShape.size());
    Ort::Value inputTensor_2 = Ort::Value::CreateTensor<float>(memoryInfo, input_2.data(), input_2.size(), inputShape.data(), inputShape.size());

    // 准备输入名称和输入张量
    const char* inputNames[] = {"input_1", "input_2"};
    std::vector<Ort::Value> inputTensors;
    inputTensors.push_back(std::move(inputTensor_1));
    inputTensors.push_back(std::move(inputTensor_2));

    // 获取输出名称
    size_t numOutputNodes = session.GetOutputCount();
    std::vector<std::string> outputNames(numOutputNodes);
    std::vector<const char*> outputNamesCStr(numOutputNodes);
    for (size_t i = 0; i < numOutputNodes; ++i) {
        Ort::AllocatedStringPtr outputName = session.GetOutputNameAllocated(i, allocator);
        outputNames[i] = outputName.get();
        outputNamesCStr[i] = outputNames[i].c_str();
    }

    // 运行推理会话
    auto outputTensors = session.Run(Ort::RunOptions{nullptr}, inputNames, inputTensors.data(), inputTensors.size(), outputNamesCStr.data(), outputNamesCStr.size());

    // 输出结果
    float* floatArray = outputTensors[0].GetTensorMutableData<float>();
    std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t outputSize = outputShape[0] * outputShape[1];

    std::cout << "Output:" << std::endl;
    for (size_t i = 0; i < outputSize; ++i) {
        std::cout << floatArray[i] << " ";
        if ((i + 1) % outputShape[1] == 0) std::cout << std::endl;
    }

    return 0;
}
