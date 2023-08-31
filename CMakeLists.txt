cmake_minimum_required(VERSION 2.8)

project( NeuralNet )

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_COMPILER g++)

set(sources 
    "src/main.cpp"
    "src/Network.cpp"
    "src/DataSet.cpp"
)

include_directories("include/")

include_directories ( "${MSYS2_DIR}/ucrt64/include/eigen3" )

add_executable( neuralNet ${sources} )

