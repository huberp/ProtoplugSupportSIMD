local ffi = require("ffi")

ffi.cdef[[
    void add_vectors(const double* a, const double* b, double* result, size_t n);
    void square_vector(const double* input, double* result, size_t n);
    double* allocate_aligned_memory(size_t n);
    void free_aligned_memory(double* ptr);
    double* compute_rms_windowed(const double* input, size_t n, size_t window);
]]

local vector_add = ffi.load("build/libvector_add")

local M = {}

local function create_aligned_memory(n)
    local ptr = vector_add.allocate_aligned_memory(n)
    if ptr == nil then
        error("Failed to allocate memory")
    end

    -- Create a metatable with a __gc metamethod to free the memory
    local mt = {
        __gc = function(cdata)
            vector_add.free_aligned_memory(cdata)
        end
    }

    -- Set the metatable for the allocated memory
    return ffi.gc(ptr, mt.__gc)
end

function M.add_vectors(a, b, n)
    local padded_n = bit.band(n + 3, bit.bnot(3))
    local result = create_aligned_memory(n)

    vector_add.add_vectors(a, b, result, padded_n)

    local result_table = {}
    for i = 0, n - 1 do
        result_table[i + 1] = result[i]
    end

    return result_table
end

function M.square_vector(input, n)
    local padded_n = bit.band(n + 3, bit.bnot(3))
    local result = create_aligned_memory(n)

    vector_add.square_vector(input, result, padded_n)

    local result_table = {}
    for i = 0, n - 1 do
        result_table[i + 1] = result[i]
    end

    return result_table
end

function M.compute_rms_windowed(input, n, window)
    local rms_values = vector_add.compute_rms_windowed(input, n, window)
    local num_windows = math.ceil(n / window)
    local result_table = {}
    for i = 0, num_windows - 1 do
        result_table[i + 1] = rms_values[i]
    end
    return result_table
end

function M.allocate_aligned_memory(n)
    return create_aligned_memory(n)
end

return M