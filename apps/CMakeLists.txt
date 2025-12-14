cmake_minimum_required(VERSION 3.10)
project(VariationalImageFiltering)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find Eigen
find_package(Eigen3 REQUIRED)

# source files
set(SOURCE_FILES image_filtering.cpp ../src/image_processing_utils.cpp)

# Create executable
add_executable(VariationalImageFiltering ${SOURCE_FILES})

# Link libraries
target_link_libraries(VariationalImageFiltering
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

# Include directories
target_include_directories(VariationalImageFiltering PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/../src
)

