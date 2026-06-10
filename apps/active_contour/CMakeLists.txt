# source files
set(ACTIVE_CONTOUR_SOURCE_FILES active_contour.cpp)

# Create executable
add_executable(ActiveContour ${ACTIVE_CONTOUR_SOURCE_FILES})

# Link libraries
target_link_libraries(ActiveContour
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

# Include directories
target_include_directories(ActiveContour PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${SRC_DIRECTORY}
)