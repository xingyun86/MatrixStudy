# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)

project ("MatrixStudy")

# Include sub-projects.
add_subdirectory ("eigen-3.3.7")
add_subdirectory ("MatrixStudy")
