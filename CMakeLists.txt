cmake_minimum_required(VERSION 3.10)
project(ProtoplugSupportSIMD)

#set(CMAKE_C_COMPILER "/mingw64/bin/gcc")

set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 -mavx -fopt-info-vec-all -S -fverbose-asm")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 /arch:AVX2")

#https://stackoverflow.com/questions/137038/how-do-you-get-assembler-output-from-c-c-source-in-gcc
#Do get access to assembly

message (${CMAKE_C_COMPILER})
message (${CMAKE_CXX_COMPILER})
#message (${CMAKE_CXX_COMPILER_AR})
#message (${CMAKE_CXX_COMPILER_RANLIB})

include_directories(${CMAKE_SOURCE_DIR}/include)

# Add the library
add_library(vector_simde_avx2 SHARED vector_simde_avx2.cpp)

# Add the executable
add_executable(main_exe main.c)

# Link the executable with the library
target_link_libraries(main_exe vector_simde_avx2)

# Add a release build configuration
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -s")
endif()