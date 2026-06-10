# source files
set(DIFFERENTIAL_GEOMETRY_SOURCE_FILES differential_geometry.cpp)

# Create executable
add_executable(DifferentialGeometry ${DIFFERENTIAL_GEOMETRY_SOURCE_FILES})

# Link libraries
target_link_libraries(DifferentialGeometry
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

# Include directories
target_include_directories(DifferentialGeometry PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${SRC_DIRECTORY}
)
