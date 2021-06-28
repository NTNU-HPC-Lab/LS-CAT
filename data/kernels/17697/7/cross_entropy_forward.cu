#include "includes.h"
extern "C" {
}
__global__ void cross_entropy_forward(unsigned int batch_size, unsigned int nclasses, const float* x, const float* t, float* y) {
int tid = blockIdx.x * blockDim.x + threadIdx.x;
if (tid < batch_size) {
// compute max value of slice
float m = x[tid*nclasses];
for(int i = 1; i < nclasses; ++i) {
m = fmaxf(x[tid*nclasses+i], m);
}
// subtract max
for(int i = 0; i < nclasses; ++i) {
y[tid*nclasses+i] = x[tid*nclasses+i]-m;
}
// sum
float s = 0.0f;
for(int i = 0; i < nclasses; ++i) {
s += expf(y[tid*nclasses+i]);
}
// compute ln(s)
float ln_s = logf(s);
// y = (ln_s - y) * t
for(int i = 0; i < nclasses; ++i) {
y[tid*nclasses+i] = (ln_s - y[tid*nclasses+i]) * t[tid*nclasses+i];
}
}
}