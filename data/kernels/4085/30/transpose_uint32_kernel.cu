#include "includes.h"
__global__ void transpose_uint32_kernel(uint32_t *src, uint32_t *dst, int src_h, int src_w, int src_align, int dst_align)
{
//l.bit_align - algined (n) by 32
//new_ldb - aligned (k) by 256
int index = blockIdx.x*blockDim.x + threadIdx.x;

//for (i = 0; i < src_h; i += 1)
int i = index % src_h;  // l.size*l.size*l.c;
{
//for (j = 0; j < src_w; j += 1)
int j = index / src_h;  // out_h*out_w;
if (j < src_w)
{
((uint32_t *)dst)[j*dst_align / 32 + i] = ((uint32_t *)src)[i*src_align + j];
}
}
}