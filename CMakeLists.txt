cmake_minimum_required(VERSION 3.10)
project(ProtoplugSupportSIMD)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx")

# Add the library
add_library(vector_add SHARED vector_add.c)

# Add the executable
add_executable(main_exe main.c)

# Link the executable with the library
target_link_libraries(main_exe vector_add)