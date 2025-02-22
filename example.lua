local ffi = require("ffi")
local vector_add = require("vector_add_ffi")

local n = 64
local a = vector_add.allocate_aligned_memory(n)
local b = vector_add.allocate_aligned_memory(n)

-- Initialize a and b with some values
for i = 0, n - 1 do
    a[i] = i
    b[i] = n - i
end

-- Call the add_vectors function
local result_add = vector_add.add_vectors(a, b, n)

-- Output the result of add_vectors to the console
for i = 1, n do
    print(string.format("result_add[%d] = %f", i - 1, result_add[i]))
end

-- Call the square_vector function
local result_square = vector_add.square_vector(a, n)

-- Output the result of square_vector to the console
for i = 1, n do
    print(string.format("result_square[%d] = %f", i - 1, result_square[i]))
end