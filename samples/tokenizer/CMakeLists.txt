cmake_minimum_required(VERSION 3.15)
project(FunGTMesher)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ════════════════════════════════════════════════════════════════════════════
# GPU BACKEND OPTIONS
# ════════════════════════════════════════════════════════════════════════════
option(FUNGT_USE_CUDA "Enable CUDA backend for NVIDIA GPUs" ON)
option(FUNGT_USE_SYCL "Enable SYCL backend for Intel GPUs" ON)
set(SYCL_CUDA_ARCH sm_75 CACHE STRING "NVIDIA GPU architecture for SYCL CUDA backend")
set_property(CACHE SYCL_CUDA_ARCH PROPERTY STRINGS sm_50 sm_52 sm_53 sm_60 sm_61 sm_62 sm_70 sm_72 sm_75 sm_80 sm_86 sm_87 sm_89 sm_90)

# ════════════════════════════════════════════════════════════════════════════
# BASE DIRECTORIES
# ════════════════════════════════════════════════════════════════════════════
if(NOT DEFINED FGT_MESHER_BASE_DIR)
    if(DEFINED ENV{FGT_MESHER_BASE_DIR})
        set(FGT_MESHER_BASE_DIR $ENV{FGT_MESHER_BASE_DIR} CACHE PATH "FunGT Mesher base directory")
        message(STATUS "Using FGT_MESHER_BASE_DIR from environment: ${FGT_MESHER_BASE_DIR}")
    else()
        message(FATAL_ERROR "FGT_MESHER_BASE_DIR not specified. Please set it via:\n"
                "  cmake -DFGT_MESHER_BASE_DIR=/path/to/fungt-mesher ..\n"
                "  or export FGT_MESHER_BASE_DIR=/path/to/fungt-mesher")
    endif()
else()
    message(STATUS "Using pre-defined FGT_MESHER_BASE_DIR: ${FGT_MESHER_BASE_DIR}")
endif()

if(NOT EXISTS "${FGT_MESHER_BASE_DIR}")
    message(FATAL_ERROR "FGT_MESHER_BASE_DIR does not exist: ${FGT_MESHER_BASE_DIR}")
endif()

# ════════════════════════════════════════════════════════════════════════════
# SYCL TOOLCHAIN DETECTION
# ════════════════════════════════════════════════════════════════════════════
if(UNIX)
    if(DEFINED SYCL_COMPILER_PATH)
        if(NOT EXISTS "${SYCL_COMPILER_PATH}")
            message(FATAL_ERROR "SYCL_COMPILER_PATH does not exist: ${SYCL_COMPILER_PATH}")
        endif()
        set(CMAKE_CXX_COMPILER ${SYCL_COMPILER_PATH})
        message(STATUS "Using user-provided SYCL compiler: ${SYCL_COMPILER_PATH}")
    else()
        set(BUNDLED_SYCL_DIR "${FGT_MESHER_BASE_DIR}/toolchain/sycl/linux_x64/dpcpp")
        set(BUNDLED_SYCL_COMPILER "${BUNDLED_SYCL_DIR}/bin/clang++")

        if(EXISTS "${BUNDLED_SYCL_COMPILER}")
            set(CMAKE_CXX_COMPILER ${BUNDLED_SYCL_COMPILER})
            message(STATUS "Using bundled SYCL compiler: ${BUNDLED_SYCL_COMPILER}")
        else()
            message(FATAL_ERROR
                "SYCL compiler not found!\n"
                "Please either:\n"
                "  1. Use bundled toolchain (should be at ${BUNDLED_SYCL_DIR})\n"
                "  2. Provide your own: cmake -DSYCL_COMPILER_PATH=/path/to/clang++ ..\n")
        endif()
    endif()
elseif(WIN32)
    if(DEFINED SYCL_COMPILER_PATH)
        if(NOT EXISTS "${SYCL_COMPILER_PATH}")
            message(FATAL_ERROR "SYCL_COMPILER_PATH does not exist: ${SYCL_COMPILER_PATH}")
        endif()
        set(CMAKE_CXX_COMPILER ${SYCL_COMPILER_PATH})
        message(STATUS "Using user-provided SYCL compiler: ${SYCL_COMPILER_PATH}")
    else()
        set(BUNDLED_SYCL_DIR "${FGT_MESHER_BASE_DIR}/toolchain/sycl/windows_x64/dpcpp")
        set(BUNDLED_SYCL_COMPILER "${BUNDLED_SYCL_DIR}/bin/clang++.exe")

        if(EXISTS "${BUNDLED_SYCL_COMPILER}")
            set(CMAKE_CXX_COMPILER ${BUNDLED_SYCL_COMPILER})
            message(STATUS "Using bundled SYCL compiler: ${BUNDLED_SYCL_COMPILER}")
        else()
            message(FATAL_ERROR
                "SYCL compiler not found!\n"
                "Please either:\n"
                "  1. Use bundled toolchain (should be at ${BUNDLED_SYCL_DIR})\n"
                "  2. Provide your own: cmake -DSYCL_COMPILER_PATH=C:/path/to/clang++.exe ..\n")
        endif()
    endif()
endif()

# ════════════════════════════════════════════════════════════════════════════
# OUTPUT DIRECTORY
# ════════════════════════════════════════════════════════════════════════════
if(UNIX)
    set(CMAKE_BUILD_TYPE Release)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/release/linux)
endif()

# ════════════════════════════════════════════════════════════════════════════
# FUNLIB
# ════════════════════════════════════════════════════════════════════════════
set(FUNLIB_DIR ${FGT_MESHER_BASE_DIR}/vendor/funlib)
add_library(funlib STATIC IMPORTED GLOBAL)
set_target_properties(funlib PROPERTIES
    IMPORTED_LOCATION ${FUNLIB_DIR}/lib/libfunlib.a
    INTERFACE_INCLUDE_DIRECTORIES ${FUNLIB_DIR}/include
)

# ════════════════════════════════════════════════════════════════════════════
# INCLUDE DIRECTORIES
# ════════════════════════════════════════════════════════════════════════════
include_directories(
    ${FGT_MESHER_BASE_DIR}
    ${FGT_MESHER_BASE_DIR}/ImageTokenizer
    ${FGT_MESHER_BASE_DIR}/vendor/stb_image
)

# ════════════════════════════════════════════════════════════════════════════
# SOURCE FILES
# ════════════════════════════════════════════════════════════════════════════
set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${FGT_MESHER_BASE_DIR}/ImageTokenizer/image_tokenizer.cpp
    ${FGT_MESHER_BASE_DIR}/vendor/stb_image/stb_image.cpp
)

# ════════════════════════════════════════════════════════════════════════════
# GPU DEVICE DETECTION BACKENDS
# ════════════════════════════════════════════════════════════════════════════
if(FUNGT_USE_CUDA)
    find_package(CUDAToolkit REQUIRED)

    if(CUDAToolkit_FOUND)
        message(STATUS "CUDA Toolkit found: ${CUDAToolkit_VERSION}")
        add_compile_definitions(FUNGT_USE_CUDA)
        enable_language(CUDA)
    else()
        message(FATAL_ERROR "CUDA requested but CUDA Toolkit not found!")
    endif()
else()
    message(STATUS "CUDA backend disabled")
endif()

# ════════════════════════════════════════════════════════════════════════════
# EXECUTABLE
# ════════════════════════════════════════════════════════════════════════════
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

if(FUNGT_USE_CUDA)
    target_compile_definitions(${PROJECT_NAME} PRIVATE FUNGT_USE_CUDA)
    target_include_directories(${PROJECT_NAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "native"
    )
endif()

# ════════════════════════════════════════════════════════════════════════════
# LINUX CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════
if(UNIX)
    message(STATUS "Configuring for Linux")

    if(FUNGT_USE_SYCL)
        set(BUNDLED_OPENCL_INCLUDE "${BUNDLED_SYCL_DIR}/include")
        find_library(BUNDLED_OPENCL_LIBRARY
            NAMES OpenCL
            PATHS "${BUNDLED_SYCL_DIR}/lib"
            NO_DEFAULT_PATH
        )

        if(EXISTS "${BUNDLED_OPENCL_INCLUDE}/CL/opencl.h" AND BUNDLED_OPENCL_LIBRARY)
            set(OpenCL_INCLUDE_DIRS "${BUNDLED_OPENCL_INCLUDE}")
            set(OpenCL_LIBRARIES "${BUNDLED_OPENCL_LIBRARY}")
            message(STATUS "OpenCL (bundled) found:")
            message(STATUS "  Include: ${OpenCL_INCLUDE_DIRS}")
            message(STATUS "  Library: ${OpenCL_LIBRARIES}")
        else()
            message(STATUS "Bundled OpenCL not found, trying system OpenCL...")
            find_package(OpenCL REQUIRED)
            if(OpenCL_FOUND)
                message(STATUS "OpenCL (system) found: ${OpenCL_INCLUDE_DIRS}")
            else()
                message(FATAL_ERROR "OpenCL library not found!")
            endif()
        endif()
    else()
        find_package(OpenCL REQUIRED)
        if(OpenCL_FOUND)
            message(STATUS "OpenCL found: ${OpenCL_INCLUDE_DIRS}")
        else()
            message(FATAL_ERROR "OpenCL library not found!")
        endif()
    endif()

    target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCL_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME}
        PRIVATE
        dl
        funlib
        ${OpenCL_LIBRARIES}
        $<$<BOOL:${FUNGT_USE_CUDA}>:CUDA::cudart_static>
        $<$<BOOL:${FUNGT_USE_CUDA}>:CUDA::cuda_driver>
    )

    # ════════════════════════════════════════════════════════════════════════
    # SYCL COMPILATION FLAGS
    # ════════════════════════════════════════════════════════════════════════
    if(CUDAToolkit_FOUND)
        message(STATUS "CUDA SYCL backend enabled")
        set(SYCL_TARGETS nvptx64-nvidia-cuda,spir64)
        set(SYCL_CUDA_ARCH_ARGS
            "-Xsycl-target-backend=nvptx64-nvidia-cuda"
            "--offload-arch=${SYCL_CUDA_ARCH}"
        )
    else()
        set(SYCL_TARGETS spir64)
        set(SYCL_CUDA_ARCH_ARGS "")
    endif()

    set(SYCL_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
        ${FGT_MESHER_BASE_DIR}/ImageTokenizer/image_tokenizer.cpp
    )

    list(JOIN SYCL_CUDA_ARCH_ARGS " " SYCL_CUDA_ARCH_STR)

    set_source_files_properties(${SYCL_SOURCES}
        PROPERTIES COMPILE_FLAGS "-fsycl -fsycl-targets=${SYCL_TARGETS} ${SYCL_CUDA_ARCH_STR}"
    )

    target_link_options(${PROJECT_NAME} PRIVATE -fsycl -fsycl-targets=${SYCL_TARGETS} ${SYCL_CUDA_ARCH_ARGS})

    if(FUNGT_USE_SYCL)
        target_compile_definitions(${PROJECT_NAME} PRIVATE FUNGT_USE_SYCL)
    endif()

endif()

# ════════════════════════════════════════════════════════════════════════════
# BUILD SUMMARY
# ════════════════════════════════════════════════════════════════════════════
message(STATUS "")
message(STATUS "════════════════════════════════════════")
message(STATUS "  ${PROJECT_NAME} Build Configuration")
message(STATUS "════════════════════════════════════════")
message(STATUS "  Compiler:         ${CMAKE_CXX_COMPILER}")
message(STATUS "  CUDA Backend:     ${FUNGT_USE_CUDA}")
message(STATUS "  SYCL Backend:     ${FUNGT_USE_SYCL}")
message(STATUS "  Build Type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "  FGT_MESHER_BASE:  ${FGT_MESHER_BASE_DIR}")
message(STATUS "════════════════════════════════════════")
message(STATUS "")
