cmake_minimum_required(VERSION 3.15)
project(imfeat LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
  add_compile_options(/O2 /arch:AVX2 /fp:fast)
else()
  add_compile_options(-O3 -march=native -ffast-math)
endif()

find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

# Google Highway: portable SIMD for the multi-channel hot loop. Header-only use
# (static dispatch, target chosen by -march); we only need its include path.
include(FetchContent)
FetchContent_Declare(
  highway
  GIT_REPOSITORY https://github.com/google/highway.git
  GIT_TAG 1.2.0
  GIT_SHALLOW TRUE)
set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(highway)

# NOMINSIZE: nanobind optimizes modules for size (-Os) by default, which the
# compiler applies AFTER our -O3 and which disables auto-vectorization of the
# SIMD hot loop (~3x slower). This module is compute-bound, so keep -O3.
nanobind_add_module(imfeat_core STABLE_ABI NOMINSIZE src/imfeat/core.cpp)
target_link_libraries(imfeat_core PRIVATE hwy)
install(TARGETS imfeat_core LIBRARY DESTINATION imfeat)