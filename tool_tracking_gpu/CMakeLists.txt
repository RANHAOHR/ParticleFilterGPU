cmake_minimum_required(VERSION 2.8.3)
project(tool_tracking_gpu)

find_package(CUDA)
find_package(OpenCV REQUIRED)
find_package(catkin REQUIRED COMPONENTS
        message_generation
        std_msgs
        sensor_msgs
        geometry_msgs
        cwru_opencv_common
        tool_model_gpu
        cwru_davinci_interface
        cwru_davinci_kinematics
        xform_utils
        )

include_directories(SYSTEM include ${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

catkin_package(CATKIN_DEPENDS message_runtime std_msgs sensor_msgs geometry_msgs
         tool_model_gpu cwru_davinci_interface cwru_davinci_kinematics cwru_opencv_common)

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




if(CUDA_FOUND AND OpenCV_CUDA_VERSION)
  cuda_add_library(tracking_particle_gpu_lib
          src/particle_filter.cu
          )

  add_executable(tracking_particle_gpu src/tracking_particle.cpp)


  target_link_libraries(tracking_particle_gpu_lib tool_model_gpu ${CUDA_LIBRARIES} davinci_kinematics davinci_interface)

  target_link_libraries(tracking_particle_gpu tracking_particle_gpu_lib)

endif()
