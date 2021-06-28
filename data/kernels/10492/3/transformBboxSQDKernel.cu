#include "includes.h"

static __device__ float E = 2.718281828;




__global__ void transformBboxSQDKernel(float *delta, float *anchor, float *res, int block_size)
{
int di = (blockIdx.x * block_size + threadIdx.x) * 4;
float d[4] = {delta[di], delta[di+1], delta[di+2], delta[di+3]};
float a[4] = {anchor[di], anchor[di+1], anchor[di+2], anchor[di+3]};
float cx = a[0] + d[0] * a[2];
float cy = a[1] + d[1] * a[3];
float w = a[2] * (d[2] < 1 ? expf(d[2]) : d[2] * E);
float h = a[3] * (d[3] < 1 ? expf(d[3]) : d[3] * E);
res[di] = cx - w * 0.5;
res[di+1] = cy - h * 0.5;
res[di+2] = cx + w * 0.5;
res[di+3] = cy + h * 0.5;
}