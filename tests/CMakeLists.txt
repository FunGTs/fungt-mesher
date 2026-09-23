cmake_minimum_required(VERSION 3.15)

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

project(FunGTMesherTests)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(FUNGT_SYCL_TARGETS spir64,nvptx64-nvidia-cuda)

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
# OPENCL
# ════════════════════════════════════════════════════════════════════════════
find_package(OpenCL REQUIRED)
if(OpenCL_FOUND)
    message(STATUS "OpenCL found")
    message(STATUS "OpenCL include dirs: ${OpenCL_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "OpenCL library not found!")
endif()

# ════════════════════════════════════════════════════════════════════════════
# INCLUDE DIRECTORIES
# ════════════════════════════════════════════════════════════════════════════
include_directories(
    ${FGT_MESHER_BASE_DIR}
    ${FGT_MESHER_BASE_DIR}/ImageTokenizer
    ${FGT_MESHER_BASE_DIR}/WeightLoader
    ${FGT_MESHER_BASE_DIR}/vendor/stb_image
)

# ════════════════════════════════════════════════════════════════════════════
# SOURCE FILES
# ════════════════════════════════════════════════════════════════════════════
set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/test_normalization.cpp
    ${FGT_MESHER_BASE_DIR}/ImageTokenizer/image_tokenizer.cpp
    ${FGT_MESHER_BASE_DIR}/vendor/stb_image/stb_image.cpp
)

# ════════════════════════════════════════════════════════════════════════════
# EXECUTABLE: Normalization Test (SYCL)
# ════════════════════════════════════════════════════════════════════════════
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} PRIVATE funlib OpenCL::OpenCL dl)
target_compile_options(${PROJECT_NAME} PRIVATE -fsycl -fsycl-targets=${FUNGT_SYCL_TARGETS})
target_link_options(${PROJECT_NAME} PRIVATE -fsycl -fsycl-targets=${FUNGT_SYCL_TARGETS})

# ════════════════════════════════════════════════════════════════════════════
# EXECUTABLE: WeightLoader Test (no SYCL required)
# ════════════════════════════════════════════════════════════════════════════
add_executable(TestWeightLoader
    ${CMAKE_CURRENT_SOURCE_DIR}/test_weight_loader.cpp
    ${FGT_MESHER_BASE_DIR}/WeightLoader/weight_loader.cpp
)

# ════════════════════════════════════════════════════════════════════════════
# BUILD SUMMARY
# ════════════════════════════════════════════════════════════════════════════
message(STATUS "")
message(STATUS "════════════════════════════════════════")
message(STATUS "  ${PROJECT_NAME} Build Configuration")
message(STATUS "════════════════════════════════════════")
message(STATUS "  Compiler:         ${CMAKE_CXX_COMPILER}")
message(STATUS "  SYCL Targets:     ${FUNGT_SYCL_TARGETS}")
message(STATUS "  Build Type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "  FGT_MESHER_BASE:  ${FGT_MESHER_BASE_DIR}")
message(STATUS "════════════════════════════════════════")
message(STATUS "")
