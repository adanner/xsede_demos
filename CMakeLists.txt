cmake_minimum_required(VERSION 2.6)

#enable supression of debug output
#add_definitions(-DNDEBUG)

#enable large file support
if (UNIX)
 set(CMAKE_CXX_FLAGS 
	 "${CMAKE_CXX_FLAGS} -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE")
 set(CMAKE_C_FLAGS 
	 "${CMAKE_C_FLAGS} -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE")
endif(UNIX)

#Turn on full warnings for GCC.
if(CMAKE_COMPILER_IS_GNUCXX)
 #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fno-stack-protector")
 set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
endif(CMAKE_COMPILER_IS_GNUCXX)

add_executable(hello hello.cpp)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

add_executable(mpi-hello mpi-hello.cpp)
target_link_libraries(mpi-hello ${MPI_LIBRARIES})
