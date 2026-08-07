cmake_minimum_required(VERSION 3.18)
project(pvsa_tensorrt_plugins LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(TENSORRT_ROOT "" CACHE PATH "TensorRT installation root")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
option(PVSA_TRT_FAST_MATH "Enable CUDA fast math for deployment performance" OFF)

find_package(CUDAToolkit REQUIRED)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
          HINTS ${TENSORRT_ROOT}/include $ENV{TENSORRT_ROOT}/include)
find_library(TENSORRT_LIBRARY nvinfer
             HINTS ${TENSORRT_ROOT}/lib ${TENSORRT_ROOT}/lib64
                   $ENV{TENSORRT_ROOT}/lib $ENV{TENSORRT_ROOT}/lib64)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
  message(FATAL_ERROR
          "找不到 TensorRT。请设置 -DTENSORRT_ROOT=/path/to/TensorRT")
endif()

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
