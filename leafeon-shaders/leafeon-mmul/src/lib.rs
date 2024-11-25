#![feature(f16)]
#![no_std]

use core::cmp::Ordering;
use core::ops::{Add, AddAssign, Mul};

use num_traits::Zero;
use spirv_std::arch::workgroup_memory_barrier_with_group_sync;
use spirv_std::glam::{UVec2, UVec3};
use spirv_std::spirv;

const TILE: usize = 32;
const TILEU32: u32 = TILE as u32;

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn mmul_tile<T: Zero + Mul<Output = T> + Add<Output = T> + Copy + AddAssign + PartialOrd>(
    local_idx: UVec3,
    global_idx: UVec3,
    mat_a: &mut [T],
    mat_b: &mut [T],
    mat_c: &mut [T],
    tile_a: &mut [[T; TILE]; TILE],
    tile_b: &mut [[T; TILE]; TILE],
    matrix_dims: &UVec3,
) {
    let n = matrix_dims.x;
    let m = matrix_dims.y;
    let p = matrix_dims.z;

    let tx = local_idx.x as usize;
    let ty = local_idx.y as usize;

    if global_idx.x < m && global_idx.y < n {
        tile_a[ty][tx] = mat_a[(global_idx.y * m + global_idx.x) as usize];
    } else {
        tile_a[ty][tx] = T::zero();
    }
    if global_idx.x < p && global_idx.y < m {
        tile_b[ty][tx] = mat_b[(global_idx.y * p + global_idx.x) as usize];
    } else {
        tile_b[ty][tx] = T::zero();
    }

    unsafe {
        workgroup_memory_barrier_with_group_sync();
    }

    let mut value = T::zero();
    for j in 0..TILE {
        value += tile_a[ty][j] * tile_b[j][tx];
    }
    unsafe {
        workgroup_memory_barrier_with_group_sync();
    }
    if global_idx.x < p && global_idx.y < m {
        mat_c[(global_idx.y * p + global_idx.x) as usize] = value;
    }
}

#[inline(always)]
#[spirv(compute(threads(32, 32)))]
pub fn mmul<T: Zero + Mul<Output = T> + Add<Output = T> + Copy>(
    #[spirv(global_invocation_id)] global_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] mat_a: &mut [T],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] mat_b: &mut [T],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] mat_c: &mut [T],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] matrix_dims: &UVec3,
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
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(global_invocation_id)] global_id: UVec3,
    #[spirv(workgroup_id)] block_idx: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] mat_a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] mat_b: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] mat_c: &mut [f32],
    #[spirv(uniform, descriptor_set = 0, binding = 3)] matrix_dims: &UVec3,
    #[spirv(workgroup)] tile_a: &mut [[f32; TILE]; TILE],
    #[spirv(workgroup)] tile_b: &mut [[f32; TILE]; TILE],
) {
    //mmul(global_id, mat_a, mat_b, mat_c, matrix_dims);
    mmul_tile(
        local_id,
        global_id,
        mat_a,
        mat_b,
        mat_c,
        tile_a,
        tile_b,
        matrix_dims,
    );
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
