cmake_minimum_required(VERSION 4.0.0)
project(Week5 VERSION 0.0.1)

include(CTest)
enable_testing()
find_package(OpenCV REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(Week5 main.cpp)

target_link_libraries(Week5 ${OpenCV_LIBS} Python3::Python)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)


