cmake_minimum_required(VERSION 3.18)
project(pvsa_tensorrt_plugins LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(TENSORRT_ROOT "" CACHE PATH "TensorRT installation root")
set(TENSORRT_INCLUDE_DIR "" CACHE PATH
    "Directory containing NvInfer.h")
set(TENSORRT_LIBRARY "" CACHE FILEPATH
    "Full path to libnvinfer.so or the TensorRT nvinfer library")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
option(PVSA_TRT_FAST_MATH "Enable CUDA fast math for deployment performance" OFF)

find_package(CUDAToolkit REQUIRED)

# 支持三种安装方式：
# 1. -DTENSORRT_ROOT=/path/to/TensorRT
# 2. 直接传入 TENSORRT_INCLUDE_DIR 和 TENSORRT_LIBRARY
# 3. Debian/Ubuntu 多架构系统目录，例如 /usr/include/x86_64-linux-gnu
if(NOT TENSORRT_INCLUDE_DIR)
  find_path(TENSORRT_INCLUDE_DIR NvInfer.h
            HINTS
              ${TENSORRT_ROOT}/include
              $ENV{TENSORRT_ROOT}/include
            PATHS
              /usr/local/TensorRT/include
              /opt/TensorRT/include
              /opt/tensorrt/include
              /usr/include/x86_64-linux-gnu
              /usr/include
              /usr/local/include)
endif()

if(NOT TENSORRT_LIBRARY)
  find_library(TENSORRT_LIBRARY
               NAMES nvinfer
               HINTS
                 ${TENSORRT_ROOT}/lib
                 ${TENSORRT_ROOT}/lib64
                 $ENV{TENSORRT_ROOT}/lib
                 $ENV{TENSORRT_ROOT}/lib64
               PATHS
                 /usr/local/TensorRT/lib
                 /usr/local/TensorRT/lib64
                 /opt/TensorRT/lib
                 /opt/TensorRT/lib64
                 /opt/tensorrt/lib
                 /opt/tensorrt/lib64
                 /usr/lib/x86_64-linux-gnu
                 /usr/lib
                 /usr/local/lib)
endif()

if(NOT TENSORRT_INCLUDE_DIR OR NOT EXISTS
   "${TENSORRT_INCLUDE_DIR}/NvInfer.h" OR NOT TENSORRT_LIBRARY OR NOT EXISTS
   "${TENSORRT_LIBRARY}")
  message(FATAL_ERROR
          "找不到完整的 TensorRT 开发文件。\n"
          "TENSORRT_ROOT=${TENSORRT_ROOT}\n"
          "TENSORRT_INCLUDE_DIR=${TENSORRT_INCLUDE_DIR}\n"
          "TENSORRT_LIBRARY=${TENSORRT_LIBRARY}\n"
          "请确认存在：<TensorRT根目录>/include/NvInfer.h 和 "
          "<TensorRT根目录>/lib(或lib64)/libnvinfer.so，或直接传入：\n"
          "-DTENSORRT_INCLUDE_DIR=/path/to/include "
          "-DTENSORRT_LIBRARY=/path/to/libnvinfer.so")
endif()

message(STATUS "TensorRT include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT library: ${TENSORRT_LIBRARY}")

add_library(pvsa_tensorrt_plugins SHARED
            src/pvsa_topp_kernel.cu
            src/pvsa_topp_plugins.cpp)

target_include_directories(pvsa_tensorrt_plugins
                           PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
                           PRIVATE ${TENSORRT_INCLUDE_DIR})
target_link_libraries(pvsa_tensorrt_plugins
                      PRIVATE ${TENSORRT_LIBRARY} CUDA::cudart)

target_compile_options(pvsa_tensorrt_plugins PRIVATE
                        $<$<COMPILE_LANGUAGE:CXX>:-O3>
                        $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
if(PVSA_TRT_FAST_MATH)
  target_compile_options(pvsa_tensorrt_plugins PRIVATE
                         $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
endif()
set_target_properties(pvsa_tensorrt_plugins PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON
                      OUTPUT_NAME pvsa_tensorrt_plugins)

install(TARGETS pvsa_tensorrt_plugins LIBRARY DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)

add_executable(pvsa_build_plugin_engine
               tools/build_plugin_engine.cpp)
target_include_directories(pvsa_build_plugin_engine
                           PRIVATE ${TENSORRT_INCLUDE_DIR})
target_link_libraries(pvsa_build_plugin_engine
                      PRIVATE pvsa_tensorrt_plugins ${TENSORRT_LIBRARY})
set_target_properties(pvsa_build_plugin_engine PROPERTIES
                      BUILD_RPATH "$ORIGIN")

install(TARGETS pvsa_build_plugin_engine RUNTIME DESTINATION bin)
