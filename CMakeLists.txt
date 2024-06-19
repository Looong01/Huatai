cmake_minimum_required(VERSION 3.13)
project(ModelInference)

# Set default ONNX Runtime path if not provided
if(NOT DEFINED ONNXRUNTIME_ROOTDIR)
    set(ONNXRUNTIME_ROOTDIR "/home/loong/.config/onnxruntime-linux-x64-1.18.0")
endif()

# Include directories
include_directories(${ONNXRUNTIME_ROOTDIR}/include)

# Link directories
link_directories(${ONNXRUNTIME_ROOTDIR}/lib)

enable_testing()

# Add executable
add_executable(huatai huatai.cpp)

# Link ONNX Runtime library
target_link_libraries(huatai onnxruntime)

add_test(NAME Test_huatai_16 COMMAND huatai ../model/fintech_model.onnx 16)