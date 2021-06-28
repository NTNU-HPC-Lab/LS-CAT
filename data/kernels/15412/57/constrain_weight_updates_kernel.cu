#include "includes.h"
__global__ void constrain_weight_updates_kernel(int N, float coef, float *weights_gpu, float *weight_updates_gpu)
{
int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if (i < N) {
const float w = weights_gpu[i];
const float wu = weight_updates_gpu[i];
const float wu_sign = (wu == 0) ? 0 : (fabs(wu) / wu);
const float abs_limit = fabs(w * coef);
if (fabs(wu) > abs_limit) weight_updates_gpu[i] = abs_limit * wu_sign;
}
}