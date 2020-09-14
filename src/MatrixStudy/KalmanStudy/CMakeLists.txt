# CMakeList.txt : CMake project for MatrixStudy, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

set(project_name KalmanStudy)
project(${project_name})

include_directories(
	${CMAKE_SOURCE_DIR}/eigen-3.3.7
)
# Add source to this project's executable.
add_executable (${project_name} "KalmanStudy.cpp")

# TODO: Add tests and install targets if needed.
