cmake_minimum_required(VERSION 3.15)

get_filename_component(FGT_MESHER_BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.."
                       ABSOLUTE)

project(ValidateUpsample LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(FUNGT_SYCL_TARGETS spir64,nvptx64-nvidia-cuda)

set(FUNLIB_DIR "${FGT_MESHER_BASE_DIR}/vendor/funlib")
add_library(funlib STATIC IMPORTED GLOBAL)
set_target_properties(funlib PROPERTIES
    IMPORTED_LOCATION "${FUNLIB_DIR}/lib/libfunlib.a"
    INTERFACE_INCLUDE_DIRECTORIES "${FUNLIB_DIR}/include")

find_package(OpenCL REQUIRED)

add_executable(validate_upsample validate_upsample.cpp)
target_link_libraries(validate_upsample PRIVATE funlib OpenCL::OpenCL GL dl)
target_compile_options(validate_upsample PRIVATE
    -fsycl -fsycl-targets=${FUNGT_SYCL_TARGETS})
target_link_options(validate_upsample PRIVATE
    -fsycl -fsycl-targets=${FUNGT_SYCL_TARGETS})
