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

include_directories("/c/Users/100053092/source/repos/matplotplusplus/source/")
link_directories("/c/Users/100053092/source/repos/matplotplusplus/build/source/matplot/")
link_directories("/c/Users/100053092/source/repos/matplotplusplus/build/source/3rd_party/")

include_directories ( "${MSYS2_DIR}/ucrt64/include/eigen3" )

include_directories("/c/Users/100053092/source/repos/msys64/ucrt64/include")
link_directories("/c/Users/100053092/source/repos/msys64/ucrt64/lib")

add_executable( neuralNet ${sources} )

target_link_libraries(neuralNet matplot nodesoup sfml-graphics.dll.a sfml-window.dll.a sfml-system.dll.a)
