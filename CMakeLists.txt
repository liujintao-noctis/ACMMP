# -----------------------------------------------------
# 1. 设置最低版本（消除警告）
# -----------------------------------------------------
cmake_minimum_required (VERSION 3.10) # 更新到 3.10+ 以消除兼容性警告

project (ACMMP CXX CUDA) # 明确指出项目使用 C++ 和 CUDA

# -----------------------------------------------------
# 2. C++ 标准升级到 C++17
# -----------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF) # 禁用 GNU 扩展，保持标准合规性

# -----------------------------------------------------
# 3. 依赖项配置 (CUDA & OpenCV)
# -----------------------------------------------------
# 注意：vcpkg 应该通过 -DCMAKE_TOOLCHAIN_FILE 传入，而不是在这里硬编码路径。
# 如果没有使用 vcpkg，请根据需要取消注释以下 CUDA 路径设置。
# set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.8")

find_package(CUDA 6.0 REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP)
find_package(OpenCV REQUIRED)

# 确保项目能够找到头文件
include_directories(${OpenCV_INCLUDE_DIRS} .)

# -----------------------------------------------------
# 4. CUDA 优化配置 (针对 GTX 1050)
# -----------------------------------------------------
# GTX 1050 是 Pascal 架构，计算能力 (compute capability) 是 6.1。
# 使用 arch=compute_61,code=sm_61 进行精确优化，并保留 sm_60 作为兼容目标。
# 移除冗余的 -std=c++11 (已被 CMAKE_CXX_STANDARD 替代)
string(APPEND CMAKE_CUDA_FLAGS " -O3 --use_fast_math --maxrregcount=128 --ptxas-options=-v --compiler-options=-Wall -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_60,code=sm_60 -std=c++17")
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)
set(CMAKE_CUDA_ARCHITECTURES 60 61)

# -----------------------------------------------------
# 5. C++ 编译标志优化
# -----------------------------------------------------
if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-pthread)
    add_definitions(-Wall)
    add_definitions(-Wextra)
    add_definitions(-pedantic)
    add_definitions(-Wno-unused-function)
    add_definitions(-Wno-switch)
    
    # 优化 Release 编译选项：使用 -O3, -march=native (针对当前 CPU 架构优化)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -ffast-math -march=native")
endif()


# -----------------------------------------------------
# 启用 CUDA 作为语言
enable_language(CUDA)

# Specify target & source files to compile it from
add_executable(
    ACMMP
    main.h
    ACMMP.h
    ACMMP.cpp
    ACMMP.cu
    main.cpp
)

# 设置源文件的语言属性
set_source_files_properties(ACMMP.cu PROPERTIES LANGUAGE CUDA)

# Specify target & libraries to link it with
target_link_libraries(ACMMP PRIVATE 
    ${OpenCV_LIBS}
    CUDA::cudart
)

if(OPENMP_FOUND)
    target_link_libraries(ACMMP PRIVATE OpenMP::OpenMP_CXX)
endif()
