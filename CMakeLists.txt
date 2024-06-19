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

# Add executable
add_executable(huatai huatai.cpp)

# Link ONNX Runtime library
target_link_libraries(huatai onnxruntime)
