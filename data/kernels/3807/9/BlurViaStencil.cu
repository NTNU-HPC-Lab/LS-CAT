#include "includes.h"
__global__ void BlurViaStencil(float* d_out, float* d_in) {
const float kernel[3][3] = {0.04, 0.12, 0.04,
0.12, 0.36, 0.12,
0.04, 0.12, 0.04};
int rowID = blockIdx.x + 1;
int colID = threadIdx.x + 1;
int pos = rowID * (blockDim.x + 2) + colID;
d_out[pos] = d_in[pos - blockDim.x - 3] * kernel[0][0]
+ d_in[pos - blockDim.x - 2] * kernel[0][1]
+ d_in[pos - blockDim.x - 1] * kernel[0][2]
+ d_in[pos - 1] * kernel[1][0]
+ d_in[pos] * kernel[1][1]
+ d_in[pos + 1] * kernel[1][2]
+ d_in[pos + blockDim.x + 1] * kernel[2][0]
+ d_in[pos + blockDim.x + 2] * kernel[2][1]
+ d_in[pos + blockDim.x + 3] * kernel[2][2];
}