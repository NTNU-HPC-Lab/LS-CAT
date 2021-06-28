#include "includes.h"
__global__ void transpose_uint32_kernel_2(uint32_t *src, uint32_t *dst, int src_h, int src_w, int src_align, int dst_align)
{
__shared__ uint32_t tmp[33 * 32];   // misaligned_array[32x32]
const int w_align = 33;
//const int shared_size = w_align * 32;

//l.bit_align - algined (n) by 32
//new_ldb - aligned (k) by 256

const int src_w_align = src_w + (32 - src_w % 32);
//const int src_h_align = src_h + (32 - src_h % 32);

const int warps_in_width = src_w_align / 32;
//const int warps_in_height = src_h_align / 32;



const int local_x = threadIdx.x % 32;   // index % 32;
const int local_x_index = threadIdx.x / 32; // index / 32;
const int local_y = local_x_index % 32;

//#pragma unroll TRANS_LOOP
//for (int i = 0; i < TRANS_LOOP; ++i)
{
const int global_index = blockIdx.x;// blockIdx.x*TRANS_LOOP + i;// local_x_index / 32;
const int global_x_index = global_index % warps_in_width;
const int global_y_index = global_index / warps_in_width;

const int global_x = global_x_index * 32 + local_x;
const int global_y = global_y_index * 32 + local_y;

uint32_t val = 0;
if (global_x < src_w && global_y < src_h) {
val = src[global_y * src_align + global_x];
}
//dst[global_x * dst_align / 32 + global_y] = val;
//tmp[local_y * 32 + local_x] = val;

tmp[local_x * w_align + local_y] = val;
__syncthreads();
val = tmp[local_y * w_align + local_x];

const int new_global_x = global_y_index * 32 + local_x;
const int new_global_y = global_x_index * 32 + local_y;

if (new_global_x < src_h && new_global_y < src_w) {
dst[new_global_y * (dst_align / 32) + new_global_x] = val;
}
}
}