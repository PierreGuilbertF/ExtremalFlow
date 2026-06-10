# source files
set(IMAGE_FILTERING_SOURCE_FILES image_filtering.cpp ../../src/image_processing_utils.cpp)

# Create executable
add_executable(VariationalImageFiltering ${IMAGE_FILTERING_SOURCE_FILES})

# Link libraries
target_link_libraries(VariationalImageFiltering
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

# Include directories
target_include_directories(VariationalImageFiltering PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${SRC_DIRECTORY}
)