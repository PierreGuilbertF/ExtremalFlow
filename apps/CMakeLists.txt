cmake_minimum_required(VERSION 3.10)
project(NumericalAnalysis)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find Eigen
find_package(Eigen3 REQUIRED)

# Source directory
set(SRC_DIRECTORY ${CMAKE_SOURCE_DIR}/../src)

add_subdirectory(image_filtering)
add_subdirectory(active_contour)
add_subdirectory(gradient_reconstruction)
add_subdirectory(differential_geometry)
