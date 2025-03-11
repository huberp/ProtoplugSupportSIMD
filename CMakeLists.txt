cmake_minimum_required(VERSION 3.10)
project(ProtoplugSupportSIMD)

#set(CMAKE_C_COMPILER "/mingw64/bin/gcc")

set(CMAKE_C_STANDARD 99)
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 -mavx -fopt-info-vec-all -S -fverbose-asm")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 -mavx /arch:AVX2")

#https://stackoverflow.com/questions/137038/how-do-you-get-assembler-output-from-c-c-source-in-gcc
#Do get access to assembly

message (${CMAKE_C_COMPILER})
message (${CMAKE_CXX_COMPILER})
#message (${CMAKE_CXX_COMPILER_AR})
#message (${CMAKE_CXX_COMPILER_RANLIB})

# Add the library
add_library(vector_simde_avx2 SHARED vector_simde_avx2.c)

# Add the executable
add_executable(main_exe main.c)

# Link the executable with the library
target_link_libraries(main_exe vector_simde_avx2)