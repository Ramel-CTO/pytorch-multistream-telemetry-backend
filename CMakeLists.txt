cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(pytorch-multistream-telemetry-backend VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Discover OpenCV dependencies on the system environment configuration maps
find_package(OpenCV REQUIRED)
# Discover Torch dependencies in the system environment library layers
find_package(Torch REQUIRED)

add_executable(telemetry_engine main.cpp)

target_include_directories(telemetry_engine PUBLIC ${OpenCV_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})
target_link_libraries(telemetry_engine ${OpenCV_LIBS} ${TORCH_LIBRARIES})
