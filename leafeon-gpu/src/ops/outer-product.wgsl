@group(0) @binding(0) var<storage, read> A: array<f32>;  // Col vector A (n)
@group(0) @binding(1) var<storage, read> B: array<f32>;  // Row vector B (m)
@group(0) @binding(2) var<storage, read_write> C: array<f32>; 

// Matrix dimensions passed from the host side
@group(0) @binding(3) var<uniform> vec_dims: vec2<u32>;  // n, m

// Compute shader entry point
@compute @workgroup_size(32, 32)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n = vec_dims.x;
    let m = vec_dims.y;

    // Get the row and column indices for the result matrix C
    let row = global_id.x;
    let col = global_id.y;

    // Check if the indices are within bounds
    if (row < n) && (col < m) {
        var sum: f32 = 0.0;

		C[row * m + col] = A[row] * B[col];
   }
}
