cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(fsrcnn_libtorh)

set(Torch_DIR /home/mint/workspace/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(src)
FILE(GLOB_RECURSE sourcefiles "src/*.cpp")
add_executable(fsrcnn ${sourcefiles})
target_link_libraries(fsrcnn ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET fsrcnn PROPERTY CXX_STANDARD 14)
