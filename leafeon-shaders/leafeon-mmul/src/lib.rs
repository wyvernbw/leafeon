#![no_std]

use core::ops::{Add, Mul};

use num_traits::Zero;
use spirv_std::glam::UVec3;
use spirv_std::spirv;

#[inline(always)]
pub fn mmul<T: Zero + Mul<Output = T> + Add<Output = T> + Copy>(
    global_id: UVec3,
    mat_a: &mut [T],
    mat_b: &mut [T],
    mat_c: &mut [T],
    matrix_dims: &UVec3,
) {
    let n = matrix_dims.x;
    let m = matrix_dims.y;
    let p = matrix_dims.z;

    let row = global_id.x;
    let col = global_id.y;

    if (row < n) && (col < p) {
        let mut sum: T = T::zero();
        for k in 0..m {
            sum = sum + mat_a[(row * m + k) as usize] * mat_b[(k * p + col) as usize];
        }
        mat_c[(row * p + col) as usize] = sum;
    }
}

#[spirv(compute(threads(32, 32)))]
pub fn mmul_f32(
    #[spirv(global_invocation_id)] global_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] mat_a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] mat_b: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] mat_c: &mut [f32],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] matrix_dims: &UVec3,
) {
    mat_a[0] += 1.0;
    mat_a[0] -= 1.0;
    mmul(global_id, mat_a, mat_b, mat_c, matrix_dims);
}

#[spirv(compute(threads(32, 32)))]
pub fn mmul_i32(
    #[spirv(global_invocation_id)] global_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] mat_a: &mut [i32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] mat_b: &mut [i32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] mat_c: &mut [i32],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] matrix_dims: &UVec3,
) {
    mmul(global_id, mat_a, mat_b, mat_c, matrix_dims);
}
