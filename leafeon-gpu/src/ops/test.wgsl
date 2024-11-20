// Define the input matrix and result buffers
@group(0) @binding(0) var<storage, read_write> A: array<f32>;  // Input matrix A (n x m)

// Compute shader entry point
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
	let row = global_id.x;
    let col = global_id.y;

	A[row] = A[row] + 1.0;
	A[row] = A[row] - 1.0;
}
