#include "includes.h"
__global__ void kernel_convert(uchar3* d_Iin, float4* d_Iout, int numel) {
size_t col = threadIdx.x + blockIdx.x * blockDim.x;
if (col >= numel) { return; }
uchar3 val = d_Iin[col];

d_Iout[col] = make_float4(
val.x / 255.0f,
val.y / 255.0f,
val.z / 255.0f,
1.0f
);
}