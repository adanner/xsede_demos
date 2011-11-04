cmake_minimum_required(VERSION 2.8)

#enable supression of debug output
#add_definitions(-DNDEBUG)

#Turn on full warnings for GCC.
if(CMAKE_COMPILER_IS_GNUCXX)
 set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
endif(CMAKE_COMPILER_IS_GNUCXX)

#standard C++ stuff
add_executable(hello hello.cpp)

#mpi stuff requires MPI
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

add_executable(mpi-hello mpi-hello.cpp)
target_link_libraries(mpi-hello ${MPI_LIBRARIES})

#cuda stuff requires CUDA, but this is optional
find_package(CUDA)
if(CUDA_FOUND)
	message("CUDA found")
	cuda_add_executable(cuda-hello cuda-hello.cu)

	cuda_add_library(vecsum cuda-vecsum.h cuda-vecsum.cu)

	add_executable(cuda-mpi cuda-mpi.cpp)
	target_link_libraries(cuda-mpi vecsum ${MPI_LIBRARIES})

	cuda_build_clean_target()
else(CUDA_FOUND)
	message("no CUDA found - CUDA apps disabled")
endif(CUDA_FOUND)


