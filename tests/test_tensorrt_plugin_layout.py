from pathlib import Path


def test_tensorrt_plugin_sources_are_self_contained():
    root = Path(__file__).resolve().parents[1]
    deploy = root / "deploy" / "tensorrt"
    kernel = (deploy / "src" / "pvsa_topp_kernel.cu").read_text(
        encoding="utf-8")
    plugin = (deploy / "src" / "pvsa_topp_plugins.cpp").read_text(
        encoding="utf-8")
    builder = (deploy / "tools" / "build_plugin_engine.cpp").read_text(
        encoding="utf-8")
    cmake = (deploy / "CMakeLists.txt").read_text(encoding="utf-8")
    readme = (deploy / "README.md").read_text(encoding="utf-8")
    root_readme = (root / "README.md").read_text(encoding="utf-8")
    deploy_readme = (root / "deploy_readme.md").read_text(encoding="utf-8")

    assert "torch::" not in kernel
    assert "torch::" not in plugin
    assert "at::" not in kernel
    assert "PYBIND11_MODULE" not in kernel
    assert "cudaStream_t stream" in kernel
    assert "cudaError_t launch_route" in kernel
    assert "cudaError_t launch_flash" in kernel
    assert '"PVSA_TopP_Route"' in plugin
    assert '"PVSA_TopP_Flash"' in plugin
    assert "IPluginV2DynamicExt" in plugin
    assert "buildSerializedNetwork" in builder
    assert "addPluginV2" in builder
    assert "getCreator" in builder
    assert "release_builder_object" in builder
    assert "pvsa_build_plugin_engine" in cmake
    assert "TENSORRT_INCLUDE_DIR" in cmake
    assert "TENSORRT_LIBRARY" in cmake
    assert "NV_TENSORRT_MAJOR < 10" in plugin
    assert "NV_TENSORRT_MAJOR < 8" in plugin
    assert "CMAKE_CUDA_HOST_COMPILER" in cmake
    assert "FP32" in readme
    assert "PVSA-Net v3.0" in root_readme
    assert "deploy_readme.md" in root_readme
    assert "PVSA-Net v3.0 部署命令" in deploy_readme
    assert "PVSA_TopP_Route" in deploy_readme
    assert "build/tensorrt/" in deploy_readme
    assert "build/tensorrt_fast/" in deploy_readme
    assert "build/tensorrt/pvsa_build_plugin_engine" in deploy_readme
    assert "build/tensorrt_fast/pvsa_build_plugin_engine" in deploy_readme
    assert "TRT_ROOT" in deploy_readme
    assert "TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz" in deploy_readme
    assert "CUDA_VISIBLE_DEVICES=1" in deploy_readme
    assert "CUDACXX=/usr/bin/nvcc" in deploy_readme
    assert "-DCMAKE_CUDA_COMPILER=/usr/bin/nvcc" in deploy_readme
    assert "/usr/local/cuda-12.0" not in deploy_readme
    assert "/usr/local/cuda-12.0" not in readme
    assert "libnvinfer.so.11.2.1" not in deploy_readme
    assert "libnvinfer.so.11.2.1" not in readme
    assert not any(line.endswith("\\\\") for line in deploy_readme.splitlines())
    assert "PVSA-v2.4" not in root_readme
    assert "PVSA-v2.4" not in deploy_readme
