cmake_minimum_required(VERSION 3.10)
project(ProtoplugSupportSIMD)

#set(CMAKE_C_COMPILER "/mingw64/bin/gcc")

set(CMAKE_C_STANDARD 17)
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 -mavx -fopt-info-vec-all -S -fverbose-asm")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 -mavx")

message (${CMAKE_C_COMPILER})
message (${CMAKE_CXX_COMPILER})
#message (${CMAKE_CXX_COMPILER_AR})
#message (${CMAKE_CXX_COMPILER_RANLIB})

# Add the library
add_library(vector_add SHARED vector_add.c)

# Add the executable
add_executable(main_exe main.c)

# Link the executable with the library
target_link_libraries(main_exe vector_add)