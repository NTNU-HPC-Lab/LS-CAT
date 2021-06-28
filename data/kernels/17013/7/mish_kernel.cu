#include "includes.h"
__device__ float softplus_kernel(float x, float threshold = 20) {
if (x > threshold) return x;                // too large
else if (x < -threshold) return expf(x);    // too small
return logf(expf(x) + 1);
}
__device__ float tanh_activate_kernel(float x){return (2/(1 + expf(-2*x)) - 1);}
__global__ void mish_kernel(const float *input, float *output, int num_elem) {

int idx = threadIdx.x + blockDim.x * blockIdx.x;
if (idx >= num_elem) return;

//float t = exp(input[idx]);
//if (input[idx] > 20.0) {
//    t *= t;
//    output[idx] = (t - 1.0) / (t + 1.0);
//} else {
//    float tt = t * t;
//    output[idx] = (tt + 2.0 * t) / (tt + 2.0 * t + 2.0);
//}
//output[idx] *= input[idx];
output[idx] = input[idx] * tanh_activate_kernel(softplus_kernel(input[idx]));
}