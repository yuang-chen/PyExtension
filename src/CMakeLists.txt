cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(grid)

set(CMAKE_PREFIX_PATH "/usr/local/include/libtorch")
find_package(Torch REQUIRED)

set(SOURCE_FILES  main.cpp grid.cpp)

add_executable(grid ${SOURCE_FILES})
target_link_libraries(grid "${TORCH_LIBRARIES}")
set_property(TARGET grid PROPERTY CXX_STANDARD 14)