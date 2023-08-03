cmake_minimum_required(VERSION 3.10)

project(ImageSimilarity)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenCV REQUIRED)

# Use pkg-config to detect libexif
find_package(PkgConfig REQUIRED)
pkg_check_modules(EXIF REQUIRED libexif)

include_directories(${OpenCV_INCLUDE_DIRS} ${EXIF_INCLUDE_DIRS})
# Link the libexif library directories
link_directories(${EXIF_LIBRARY_DIRS})

add_executable(ImageSimilarity src/similarity.cpp)

target_link_libraries(ImageSimilarity ${OpenCV_LIBS} ${EXIF_LIBRARIES})
