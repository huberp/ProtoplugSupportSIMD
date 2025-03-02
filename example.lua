local ffi = require("ffi")
local vector_add = require("vector_add_ffi")

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
for i = 0, n - 1 do
    print(string.format("a[%d] = %d; b[%d] = %d", i, _a[i], i, _b[i]))
end
local c = vector_add.allocate_aligned_memory(n)
local d = vector_add.allocate_aligned_memory(n)
local e = vector_add.allocate_aligned_memory(n)
print("=====================================")
print("=====================================")
for i = 0, n - 1 do
    print(string.format("a[%d] = %d; b[%d] = %d", i, _a[i], i, _b[i]))
end
--error("stop")

-- Call the add_vectors function
local result_add = vector_add.add_vectors(a, b, n)
local _result_add = result_add()
print("result_add: ", result_add)
-- Output the result of add_vectors to the console
for i = 1, n-1 do
    print(string.format("a[%d] = %d; b[%d] = %d --> result_add[%d] = %f", i, _a[i], i, _b[i], i, _result_add[i]))
end
result_add = nil
_result_add = nil

-- Call the square_vector function
local result_square = vector_add.square_vector(a, n)
local _result_square = result_square()  
-- Output the result of square_vector to the console
for i = 1, n do
    print(string.format("result_square[%d] = %f", i - 1, _result_square[i]))
end
result_square = nil
_result_square = nil
collectgarbage("collect")

-- now show how the windowed rms works
local result_rms_windows = vector_add.compute_rms_windowed(a, n, 12)
for i = 1, #result_rms_windows do
    print(string.format("result_rms[%d] = %f", i - 1, result_rms_windows[i]))
end