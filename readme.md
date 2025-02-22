
### Building the DLL

1. Open a terminal and navigate to the project directory:
    ```sh
    cd /x:/dev/projects/ProtoplugScripts
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

This will generate a `vector_add.dll` file in the `build` directory. You can then use this DLL in your Lua script or any other application that supports loading DLLs.