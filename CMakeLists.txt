cmake_minimum_required (VERSION 3.13)
project(EdgeDirectVO LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
#set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall")

find_package(OpenCV 4 REQUIRED)
find_package (Eigen3 3.3 REQUIRED CONFIG)
find_package(Boost REQUIRED)

#message(STATUS "Path is: ’${OpenCV_PATH}'")
message("\n=====OpenCV Library=====")
message(STATUS "[Directory]:    ${OpenCV_DIR}")
message(STATUS "[Version]:      ${OpenCV_VERSION}")
message(STATUS "[Libraries]:    ${OpenCV_LIBS}")
message(STATUS "[Include Path]: ${OpenCV_INCLUDE_DIRS}")
message( "========================\n")

include_directories(${EdgeDirectVO_SOURCE_DIR}/include
                    ${EdgeDirectVO_SOURCE_DIR}/Eigen)


file(GLOB SOURCES "include/*.h" "src/*.cpp")
add_executable(EdgeDirectVO ${SOURCES})

target_link_libraries (EdgeDirectVO  ${OpenCV_LIBS} 
                                     Eigen3::Eigen)
                                    #"-lopencv_ximgproc")         
  