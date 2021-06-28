#include "includes.h"
__global__ void NN_UpSampling( float *target, const float *source, const int wt, const int ht )
{
int y = blockIdx.y * blockDim.y + threadIdx.y;
int x = blockIdx.x * blockDim.x + threadIdx.x;
const int curt = y*wt+x;
const int curs = (y/2)*(wt/2)+x/2;
if(y < ht and x < wt) {
target[curt*3+0] = source[curs*3+0];
target[curt*3+1] = source[curs*3+1];
target[curt*3+2] = source[curs*3+2];
}
}