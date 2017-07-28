    cmake_minimum_required(VERSION 2.8.3)
project(tool_model_gpu)

#uncomment next line to use OpenCV library
find_package(OpenCV REQUIRED)
find_package(CUDA)
find_package(catkin REQUIRED roscpp cwru_opencv_common)


#uncomment the next line to use the point-cloud library
#find_package(PCL 1.7 REQUIRED)

#uncomment the following 4 lines to use the Eigen library
#find_package(cmake_modules REQUIRED)
#find_package(Eigen3 REQUIRED)
#include_directories(${EIGEN3_INCLUDE_DIR})
#add_definitions(${EIGEN_DEFINITIONS})
include_directories(
        include
        ${catkin_INCLUDE_DIRS}
)


catkin_package(CATKIN_DEPENDS std_msgs sensor_msgs cwru_opencv_common)

catkin_package(
        DEPENDS EIGEN_DEP
        LIBRARIES tool_model_gpu
        INCLUDE_DIRS include
)

if(CUDA_FOUND)
    message("...Found CUDA ${CUDA_VERSION}")
    if(OpenCV_CUDA_VERSION)
        message("...Opencv compiled with cuda version ${OpenCV_CUDA_VERSION}")
    endif()
endif()


# If necessary set the nvidia build flags for the GPU architecture.
if(CUDA_FOUND AND OpenCV_CUDA_VERSION)
    SET(CUDA_NVCC_FLAGS "-arch=compute_30" CACHE STRING "nvcc flags" FORCE)
endif()
# example boost usage
# find_package(Boost REQUIRED COMPONENTS system thread)

# C++0x support - not quite the same as final C++11!
# use carefully;  can interfere with point-cloud library
# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -pg -Q" )
# SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg -Q")

#the following is required, if desire to link a node in this package with a library created in this same package
# edit the arguments to reference the named node and named library within this package
# target_link_library(example my_lib)

if(CUDA_FOUND AND OpenCV_CUDA_VERSION)
    cuda_add_library(tool_model_gpu src/tool_model.cu)

    add_executable(tool_model_gpu_test src/tool_model_main.cpp)

    #the following is required, if desire to link a node in this package with a library created in this same package
    # edit the arguments to reference the named node and named library within this package
    # target_link_library(example my_lib)
    target_link_libraries(tool_model_gpu ${OpenCV_LIBRARIES} ${catkin_LIBRARIES} ${CUDA_LIBRARIES} )
    target_link_libraries(tool_model_gpu_test tool_model_gpu)

endif()
