#include "includes.h"
__global__ void kSoftmaxGrad(float* dE_dy_l, float* y_l, float* dE_dx_l, int numCases, int numOut) {
const int tx = blockIdx.x * LOGREG_GRAD_THREADS_X + threadIdx.x;
const int ty = blockIdx.y * LOGREG_GRAD_THREADS_Y + threadIdx.y;
const int tidx = ty * numCases + tx;

if (ty < numOut && tx < numCases) {
float v = 0;
for (int j = 0; j < numOut; j++) {
v += dE_dy_l[j * numCases + tx] * ((j == ty) - y_l[j * numCases + tx]);
}
v *= y_l[tidx];
dE_dx_l[tidx] = v;
}
}