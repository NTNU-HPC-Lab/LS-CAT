#include "includes.h"
__global__ void momentum_update_1D_1D(float* x, float* d, float* m, float learning_rate, float momentum, float gradClip, bool nesterov, int size) {
int tid = blockIdx.x * blockDim.x + threadIdx.x;
int stride = gridDim.x * blockDim.x;
for (; tid < size; tid += stride) {
float temp = d[tid];
if (temp > gradClip) temp = gradClip;
if (temp < -gradClip) temp = -gradClip;
m[tid] *= momentum;
m[tid] += temp;
if (nesterov) { temp += momentum * m[tid]; }
else { temp = m[tid]; }
x[tid] -= learning_rate * temp;
d[tid] = 0;
}
}