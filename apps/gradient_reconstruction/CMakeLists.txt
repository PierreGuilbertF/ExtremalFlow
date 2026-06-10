# source files
set(GRADIENT_RECONSTRUCTION_SOURCE_FILES gradient_reconstruction.cpp)

# Create executable
add_executable(GradientReconstruction ${GRADIENT_RECONSTRUCTION_SOURCE_FILES})

# Link libraries
target_link_libraries(GradientReconstruction
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

# Include directories
target_include_directories(GradientReconstruction PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${SRC_DIRECTORY}
)