#include <iostream>
#include <vector>
#include <random>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

std::vector<double> generateRandomMatrix(size_t rows, size_t cols) {
    std::vector<double> matrix(rows * cols);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (size_t i = 0; i < rows * cols; ++i) {
        matrix[i] = static_cast<double>(dis(gen));
    }
    return matrix;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
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

    // Generate random inputs
    std::vector<double> input_1 = generateRandomMatrix(batchSize, inputDim);
    std::vector<double> input_2 = generateRandomMatrix(batchSize, inputDim);

    // Create input tensors
    std::vector<int64_t> inputShape = {batchSize, static_cast<int64_t>(inputDim)};
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value inputTensor_1 = Ort::Value::CreateTensor<double>(memoryInfo, input_1.data(), input_1.size(), inputShape.data(), inputShape.size());
    Ort::Value inputTensor_2 = Ort::Value::CreateTensor<double>(memoryInfo, input_2.data(), input_2.size(), inputShape.data(), inputShape.size());

    // Prepare input names and input tensors
    const char* inputNames[] = {"input_1", "input_2"};
    std::vector<Ort::Value> inputTensors;
    inputTensors.push_back(std::move(inputTensor_1));
    inputTensors.push_back(std::move(inputTensor_2));

    // Get output names
    size_t numOutputNodes = session.GetOutputCount();
    std::vector<std::string> outputNames(numOutputNodes);
    std::vector<const char*> outputNamesCStr(numOutputNodes);
    for (size_t i = 0; i < numOutputNodes; ++i) {
        Ort::AllocatedStringPtr outputName = session.GetOutputNameAllocated(i, allocator);
        outputNames[i] = outputName.get();
        outputNamesCStr[i] = outputNames[i].c_str();
    }

    // Run inference
    auto outputTensors = session.Run(Ort::RunOptions{nullptr}, inputNames, inputTensors.data(), inputTensors.size(), outputNamesCStr.data(), outputNamesCStr.size());

    // Get the output tensor and print results
    double* doubleArray = outputTensors[0].GetTensorMutableData<double>();
    std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t outputSize = outputShape[0] * outputShape[1];

    std::cout << "Output:" << std::endl;
    for (size_t i = 0; i < outputSize; ++i) {
        std::cout << doubleArray[i] << " ";
        if ((i + 1) % outputShape[1] == 0) std::cout << std::endl;
    }

    return 0;
}
