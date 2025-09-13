# Vector Simd Lib 

## xsimd
This has been taken from the header only lib xsimd, release v0.8.4
https://github.com/xtensor-stack/xsimd

only the subdirectory "esimd" has been copied into this project.
the subdirectories esimd/wasm and esimd/mips have been removed 

## Building the DLL

Requirement: cmake and compiler infrastructure is available.
For instance msys2 in case of windows. 

1. Open a terminal and navigate to the project directory:
    ```sh
    cd <projectroot>/ProtoplugSupportSIMD
    ```

2. Create a `build` directory and navigate into it:
    ```sh
    mkdir build
    cd build
    ```

3. Run CMake to generate the build files:
    ```sh
    cmake -G "Unix Makefiles" ..
    ```

4. Build the DLL:
    ```sh
    cmake --build .
    ```

This will generate a `vector_simde_avx2.dll` file in the `build` directory. You can then use this DLL in your Lua script or any other application that supports loading DLLs.
Find the lua binding in <projectroot>/vector_simd.lua

## Run in Console
There's a "<projectroot>/main.c" which compiles into main.exe (build dir). It shows how the functions in the DLL can be used.

## Run with lua
Prerequisite: luajit has been installed. Run "<projectroot>/example.lua". It shows how to use the vetor_simd.lua

cd into root directory
```
luajit example.lua
```
