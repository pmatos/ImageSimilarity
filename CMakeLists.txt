cmake_minimum_required(VERSION 3.10)

project(ImageSimilarity)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Replace the path with your own path to OpenCV
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(ImageSimilarity src/similarity.cpp)

target_link_libraries(ImageSimilarity ${OpenCV_LIBS})
