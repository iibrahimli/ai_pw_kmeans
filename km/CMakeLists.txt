cmake_minimum_required(VERSION 3.10)

project(km VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${PROJECT_SOURCE_DIR}/include)

set(TARGET_EXEC main)

add_executable(${TARGET_EXEC} ${PROJECT_SOURCE_DIR}/src/main.cpp)

# use OpenMP if available
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${TARGET_EXEC} OpenMP::OpenMP_CXX)
endif()
