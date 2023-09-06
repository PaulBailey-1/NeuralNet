cmake_minimum_required(VERSION 2.8)

project( NeuralNet )

# add_subdirectory(../matplotplusplus ../../matplotplusplus/build)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(sources 
    "src/main.cpp"
    "src/Network.cpp"
    "src/DataSet.cpp"
)

include_directories("include/")

find_package(Matplot++ REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_executable( neuralNet ${sources} )

target_link_libraries(neuralNet PUBLIC Matplot++::matplot sfml-system sfml-graphics sfml-window PRIVATE Threads::Threads)
