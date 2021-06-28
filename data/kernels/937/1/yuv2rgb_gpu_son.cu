#include "includes.h"


__device__ unsigned char clip_rgb_gpu(int x)
{
if(x > 255)
return 255;
if(x < 0)
return 0;

return (unsigned char)x;
}
__global__ void yuv2rgb_gpu_son(unsigned char * d_y , unsigned char * d_u ,unsigned char * d_v , unsigned char * d_r, unsigned char * d_g, unsigned char * d_b, int size)
{
int x = threadIdx.x + blockDim.x*blockIdx.x;
if (x >= size) return;
int rt,gt,bt;
int y, cb, cr;

y  = ((int)d_y[x]);
cb = ((int)d_u[x]) - 128;
cr = ((int)d_v[x]) - 128;

rt  = (int)( y + 1.402*cr);
gt  = (int)( y - 0.344*cb - 0.714*cr);
bt  = (int)( y + 1.772*cb);

d_r[x] = clip_rgb_gpu(rt);
d_g[x] = clip_rgb_gpu(gt);
d_b[x] = clip_rgb_gpu(bt);
}