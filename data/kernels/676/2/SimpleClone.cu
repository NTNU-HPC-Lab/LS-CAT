#include "includes.h"
__global__ void SimpleClone( const float *background, const float *target, const float *mask, float *output, const int wb, const int hb, const int wt, const int ht, const int oy, const int ox )
{
const int yt = blockIdx.y * blockDim.y + threadIdx.y;
const int xt = blockIdx.x * blockDim.x + threadIdx.x;
const int curt = wt*yt+xt;
if (yt < ht and xt < wt and mask[curt] > 127.0f) {
const int yb = oy+yt, xb = ox+xt;
const int curb = wb*yb+xb;
if (0 <= yb and yb < hb and 0 <= xb and xb < wb) {
output[curb*3+0] = target[curt*3+0];
output[curb*3+1] = target[curt*3+1];
output[curb*3+2] = target[curt*3+2];
}
}
}