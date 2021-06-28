#include "includes.h"
__global__ void Shrink_DownSampling( float *target, const float *source, const int wt, const int ht, const int ws, const int hs )
{
int y = blockIdx.y * blockDim.y + threadIdx.y;
int x = blockIdx.x * blockDim.x + threadIdx.x;
const int curt = y*wt+x;
const int curs = (y*2)*ws+x*2;
if(y < ht and x < wt) {
target[curt*3+0] = (source[curs*3+0]+source[(curs+1)*3+0]+source[(curs+ws)*3+0]+source[(curs+ws+1)*3+0])/4.0f;
target[curt*3+1] = (source[curs*3+1]+source[(curs+1)*3+1]+source[(curs+ws)*3+1]+source[(curs+ws+1)*3+1])/4.0f;
target[curt*3+2] = (source[curs*3+2]+source[(curs+1)*3+2]+source[(curs+ws)*3+2]+source[(curs+ws+1)*3+2])/4.0f;
}
}