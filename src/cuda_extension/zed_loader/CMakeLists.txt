cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(zed_loader LANGUAGES CXX)

# ============================================================================
# Build Configuration
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Sanitizers for Debug builds (AddressSanitizer + UndefinedBehaviorSanitizer)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Enabling AddressSanitizer and UndefinedBehaviorSanitizer for Debug build")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address,undefined -fno-omit-frame-pointer")
    set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -fsanitize=address,undefined")
    set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} -fsanitize=address,undefined")
endif()

# ============================================================================
# Dependencies
# ============================================================================

# Threading support
find_package(Threads REQUIRED)

# Option to build the demo executable (requires Rerun SDK)
option(BUILD_ZEDLOADER_EXEC "Build the ZedLoader demo executable (requires Rerun SDK)" ON)

# Python - use Development.Module for pybind11 compatibility
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# PyTorch / ATen — manual setup to bypass find_package(Torch), which internally
# calls enable_language(CUDA) and requires nvcc. Since ZedLoader only compiles
# C++ and links against Torch/CUDA libraries, we query paths from Python directly.
execute_process(
    COMMAND ${Python_EXECUTABLE} -c
        "import torch, os; d=os.path.join(os.path.dirname(torch.__file__),'include'); print(';'.join([d, os.path.join(d,'torch','csrc','api','include')]))"
    OUTPUT_VARIABLE TORCH_INCLUDE_DIRS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
execute_process(
    COMMAND ${Python_EXECUTABLE} -c
        "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"
    OUTPUT_VARIABLE TORCH_LIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
execute_process(
    COMMAND ${Python_EXECUTABLE} -c
        "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
    OUTPUT_VARIABLE _TORCH_CXX11_ABI
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=${_TORCH_CXX11_ABI})
link_directories(${TORCH_LIB_DIR})
set(TORCH_LIBRARIES torch torch_cpu c10)

# pybind11
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_CMAKE_DIR})
find_package(pybind11 REQUIRED)

# ZED SDK
find_package(ZED 4 REQUIRED)

# CUDA
find_package(CUDAToolkit REQUIRED)

# ============================================================================
# Library Sources
# ============================================================================
set(ZEDLOADER_SOURCES
    src/zed_loader.cpp
)

set(ZEDLOADER_HEADERS
    include/zed_loader/zed_loader.hpp
    include/zed_loader/common.hpp
)

# ============================================================================
# Core Library
# ============================================================================
add_library(zed_loader_core SHARED
    ${ZEDLOADER_SOURCES}
    ${ZEDLOADER_HEADERS}
)

target_include_directories(zed_loader_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Mark external library includes as SYSTEM to suppress warnings
target_include_directories(zed_loader_core SYSTEM
    PUBLIC
        ${ZED_INCLUDE_DIRS}
        ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(zed_loader_core
    PUBLIC
        ${ZED_LIBRARIES}
        ${TORCH_LIBRARIES}
        CUDA::cudart
        Threads::Threads
)

target_compile_features(zed_loader_core PUBLIC cxx_std_17)

# Enable ATen parallel operations
target_compile_definitions(zed_loader_core PRIVATE
    AT_PARALLEL_OPENMP=1
)

# ============================================================================
# Python Bindings
# ============================================================================
pybind11_add_module(zed_loader_py
    python/bindings.cpp
)

target_include_directories(zed_loader_py
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Mark external library includes as SYSTEM to suppress warnings
target_include_directories(zed_loader_py SYSTEM
    PRIVATE
        ${ZED_INCLUDE_DIRS}
        ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(zed_loader_py
    PRIVATE
        zed_loader_core
        ${TORCH_LIBRARIES}
        torch_python
)

# Set output name for python module
set_target_properties(zed_loader_py PROPERTIES
    OUTPUT_NAME "zed_loader"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

# ============================================================================
# Executable for testing (optional, requires Rerun SDK)
# ============================================================================
if(BUILD_ZEDLOADER_EXEC)
    # Rerun SDK for visualization/logging (only needed for executable)
    add_subdirectory(/usr/local/rerun_cpp_sdk ${CMAKE_BINARY_DIR}/rerun_sdk)

    add_executable(
        zed_loader_exec
        src/main.cpp
    )
    target_link_libraries(zed_loader_exec
        PRIVATE
            zed_loader_core
            rerun_sdk
    )
    target_compile_features(zed_loader_exec PRIVATE cxx_std_17)
endif()

# ============================================================================
# Installation
# ============================================================================
install(TARGETS zed_loader_core
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
)

# ============================================================================
# Summary
# ============================================================================
message(STATUS "")
message(STATUS "=== ZedLoader Build Configuration ===")
message(STATUS "  CMAKE_BUILD_TYPE:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  CMAKE_CXX_STANDARD:   ${CMAKE_CXX_STANDARD}")
message(STATUS "  Python:               ${Python_EXECUTABLE}")
message(STATUS "  PyTorch includes:     ${TORCH_INCLUDE_DIRS}")
message(STATUS "  PyTorch lib:          ${TORCH_LIB_DIR}")
message(STATUS "  pybind11:             ${PYBIND11_CMAKE_DIR}")
message(STATUS "  ZED SDK:              ${ZED_INCLUDE_DIRS}")
message(STATUS "  CUDA:                 ${CUDAToolkit_VERSION}")
message(STATUS "  Build executable:     ${BUILD_ZEDLOADER_EXEC}")
if(BUILD_ZEDLOADER_EXEC)
    message(STATUS "  Rerun SDK:            /usr/local/rerun_cpp_sdk")
endif()
message(STATUS "======================================")
message(STATUS "")

