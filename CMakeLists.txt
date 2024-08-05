cmake_minimum_required(VERSION 3.10)
project(CudaProgramming)

# Set the CUDA toolkit path if not found automatically
if(NOT DEFINED ENV{CUDA_TOOLKIT_ROOT_DIR})
    set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
endif()

# Find CUDA package
find_package(CUDA REQUIRED)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# # Include directories
include_directories(
    # ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Add source files
file(GLOB CUDA_SOURCES ${PROJECT_SOURCE_DIR}/src/cuda/*.cu)
file(GLOB CPP_SOURCES ${PROJECT_SOURCE_DIR}/src/csrc/*.cpp)
file(GLOB CPU_SOURCES ${PROJECT_SOURCE_DIR}/src/cpu/*.cpp)

# Create executable
cuda_add_executable(CudaProgramming ${CUDA_SOURCES} ${CPP_SOURCES} ${CPU_SOURCES})

# Specify include directories for the target
# target_include_directories(MyProject PRIVATE
#     ${PROJECT_SOURCE_DIR}/src/cuda
#     ${PROJECT_SOURCE_DIR}/src/csrc
#     ${PROJECT_SOURCE_DIR}/src/cpu
# )

# Custom clean target
add_custom_target(clean_build
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_BINARY_DIR}/clean_build.cmake
    COMMENT "Cleaning the build directory"
)

# Add a custom command to remove the build directory
file(WRITE ${CMAKE_BINARY_DIR}/clean_build.cmake
"file(REMOVE_RECURSE ${CMAKE_BINARY_DIR}/CMakeFiles)
file(REMOVE ${CMAKE_BINARY_DIR}/CMakeCache.txt)
file(REMOVE ${CMAKE_BINARY_DIR}/cmake_install.cmake)
file(REMOVE ${CMAKE_BINARY_DIR}/Makefile)
")

# Custom target to perform a fresh build
add_custom_target(fresh_build
    COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_SOURCE_DIR} make clean_build
    COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_SOURCE_DIR} cmake -B${CMAKE_BINARY_DIR} -H${CMAKE_SOURCE_DIR}
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
    COMMENT "Starting a fresh build"
    DEPENDS clean_build
)

# Link libraries if necessary
# target_link_libraries(MyProject some_library)
target_link_libraries(CudaProgramming ${OpenCV_LIBS})