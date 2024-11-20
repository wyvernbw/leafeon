// Define the input matrix and result buffers
@group(0) @binding(0) var<storage_buffer> A: array<f32>;  // Input matrix A (n x m)
@group(0) @binding(1) var<storage_buffer> B: array<f32>;  // Input matrix B (m x p)
@group(0) @binding(2) var<storage_buffer> C: array<f32>;  // Output matrix C (n x p)

// Matrix dimensions passed from the host side
@group(0) @binding(3) var<uniform> matrix_dims: vec3<u64>;  // n, m, p

// Compute shader entry point
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n = matrix_dims.x;
    let m = matrix_dims.y;
    let p = matrix_dims.z;

    // Get the row and column indices for the result matrix C
    let row = global_id.x;
    let col = global_id.y;

    // Check if the indices are within bounds
    if (row < n) && (col < p) {
        var sum: f32 = 0.0;

        // Perform the dot product of row from A and column from B
        for (var k = 0u; k < m; k = k + 1u) {
            sum = sum + A[row * m + k] * B[k * p + col];
        }

        // Store the result in C
        C[row * p + col] = sum;
    }
}
