#include "includes.h"
__global__ static void yuv422_to_yuv444_kernel(const void * src, void * out, int pix_count) {
// coordinates of this thread
const int block_idx_x = threadIdx.x + blockIdx.x * blockDim.x;

// skip if out of bounds
if(block_idx_x >= pix_count / 2) {
return;
}

uchar4 *this_src = ((uchar4 *) src) + block_idx_x * 2;
uchar4 *this_out = ((uchar4 *) out) + block_idx_x * 3;

uchar4 pix12 = this_src[0];
uchar4 pix34 = this_src[1];

uchar4 out_pix[3];
out_pix[0].x = pix12.y;
out_pix[0].y = pix12.x;
out_pix[0].z = pix12.z;

out_pix[0].w = pix12.w;
out_pix[1].x = pix12.x;
out_pix[1].y = pix12.z;

out_pix[1].z = pix34.y;
out_pix[1].w = pix34.x;
out_pix[2].x = pix34.z;

out_pix[2].y = pix34.w;
out_pix[2].z = pix34.x;
out_pix[2].w = pix34.z;

this_out[0] = out_pix[0];
this_out[1] = out_pix[1];
this_out[2] = out_pix[2];
}