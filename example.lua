local ffi = require("ffi")
local vector_add = require("vector_simd")

-- Example usage of compute_abs_ratio
local function example_compute_abs_ratio()
    -- Define the size of the vectors
    local n = 8

    -- Allocate aligned memory for the input and output vectors
    local a, _ = vector_add.allocate_aligned_memory(n)
    local b, _ = vector_add.allocate_aligned_memory(n)
    local result, _ = vector_add.allocate_aligned_memory(n)

    -- Initialize the input vectors with some values
    local j = 0 -- gives sequence 0,1,0,1,0,1,0,1
    local fac = -1
    for i = 0, n - 1 do
        j=1-j
        fac = -1 + (2*j) -- gives sequence -1,1,-1,1,-1,1,-1,1
        a()[i] = i + 1
        b()[i] = (i + 1) * 2 * fac
    end

    -- Compute the abs(a + b) / (abs(a) + abs(b)) ratio
    vector_add.compute_abs_ratio_into(a, b, result, n)

    -- Print the result
    for i = 0, n - 1 do
        print(string.format("result[%d] = %f", i, result()[i]))
    end
end

-- Call the example function
example_compute_abs_ratio()

-- Example usage of compute_abs_ratio
local function example_add()
    local n = 256
    local a = vector_add.allocate_aligned_memory(n)
    local b = vector_add.allocate_aligned_memory(n)

    -- Initialize a and b with some values
    local _a = a()
    local _b = b()
    for i = 0, n - 1 do
        _a[i] = i
        _b[i] = n - i
    end

    -- Call the add_vectors function
    local result_add = vector_add.add_vectors(a, b, n)
    local _result_add = result_add()
    print("result_add: ", result_add)
    -- Output the result of add_vectors to the console
    for i = 1, n-1 do
        print(string.format("a[%d] = %d; b[%d] = %d --> result_add[%d] = %f", i, _a[i], i, _b[i], i, _result_add[i]))
    end
end

example_add()

local function example_square()
    local n = 256
    local a = vector_add.allocate_aligned_memory(n)
    local b = vector_add.allocate_aligned_memory(n)

    -- Initialize a and b with some values
    local _a = a()
    local _b = b()
    for i = 0, n - 1 do
        _a[i] = i
        _b[i] = n - i
    end

    -- Call the square_vector function
    local result_square = vector_add.square_vector(a, n)
    local _result_square = result_square()  
    -- Output the result of square_vector to the console
    for i = 1, n do
        print(string.format("result_square[%d] = %f", i - 1, _result_square[i]))
    end
end

example_square()

local function example_rms_window()
    local n = 256
    local a = vector_add.allocate_aligned_memory(n)
    local b = vector_add.allocate_aligned_memory(n)

    -- Initialize a and b with some values
    local _a = a()
    local _b = b()
    for i = 0, n - 1 do
        _a[i] = i
        _b[i] = n - i
    end

    -- now show how the windowed rms works
    local result_rms_windows = vector_add.compute_rms_windowed(a, n, 12)
    for i = 1, #result_rms_windows do
        print(string.format("result_rms[%d] = %f", i - 1, result_rms_windows[i]))
    end
end

example_rms_window()

local function example_squared_difference()
    local n = 8
    local a = vector_add.allocate_aligned_memory(n)
    local b = vector_add.allocate_aligned_memory(n)
    local result = vector_add.allocate_aligned_memory(n)

    -- Initialize a and b with some values
    local _a = a()
    local _b = b()
    local j = 0 -- gives sequence 0,1,0,1,0,1,0,1
    local fac = -1
    for i = 0, n - 1 do
        j=1-j
        fac = -1 + (2*j) -- gives sequence -1,1,-1,1,-1,1,-1,1
        _a[i] = i + 1
        _b[i] = (i + 1) * 2 * fac
    end

    -- Call the squared_difference function
    vector_add.squared_difference_into(a, b, result, n)
    local _result = result()

    -- Output the result of squared_difference to the console
    for i = 0, n - 1 do
        print(string.format("squared_difference[%d] = %f", i, _result[i]))
    end
end

example_squared_difference()

local function example_compute_a_plus_bx()
    local n = 1024
    local a = 10000.0
    local b = 0.5
    local x = vector_add.allocate_aligned_memory(n)
    local result = vector_add.allocate_aligned_memory(n)

    -- Initialize x with some values
    local _x = x()
    for i = 0, n - 1 do
        _x[i] = i
    end

    -- Call the compute_a_plus_bx function
    vector_add.compute_a_plus_bx_into(a, b, x, result, n)
    local _result = result()

    -- Output the result of compute_a_plus_bx to the console
    for i = 0, n - 1 do
        print(string.format("compute_a_plus_bx[%d] = %f", i, _result[i]))
    end
end

example_compute_a_plus_bx()