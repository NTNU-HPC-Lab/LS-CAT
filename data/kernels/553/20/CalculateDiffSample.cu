#include "includes.h"
__global__ void CalculateDiffSample( float *cur, float *pre, const int wts, const int hts ){
const int yts = blockIdx.y * blockDim.y + threadIdx.y;
const int xts = blockIdx.x * blockDim.x + threadIdx.x;
const int curst = wts * yts + xts;

if (yts < hts && xts < wts){
cur[curst*3+0] -= pre[curst*3+0];
cur[curst*3+1] -= pre[curst*3+1];
cur[curst*3+2] -= pre[curst*3+2];
pre[curst*3+0] = 0;
pre[curst*3+1] = 0;
pre[curst*3+2] = 0;
}
}