#include "includes.h"
__device__ float softplus_kernel(float x, float threshold = 20) {
if (x > threshold) return x;                // too large
else if (x < -threshold) return expf(x);    // too small
return log1pf(expf(x));
//return logf(expf(x) + 1);
}
__global__ void gradient_array_mish_kernel(int n, float *activation_input_gpu, float *delta)
{
int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if (i < n) {
const float MISH_THRESHOLD = 20.0f;

// implementation from TensorFlow: https://github.com/tensorflow/addons/blob/093cdfa85d334cbe19a37624c33198f3140109ed/tensorflow_addons/custom_ops/activations/cc/kernels/mish_op.h#L66-L80
// implementation from Pytorch: https://github.com/thomasbrandon/mish-cuda/blob/master/csrc/mish.h#L26-L31
// log1p(x) == log(x + 1)
const float inp = activation_input_gpu[i];
const float sp = softplus_kernel(inp, MISH_THRESHOLD);
const float grad_sp = -expm1f(-sp);
//const float grad_sp = 1 - expf(-sp);
const float tsp = tanh(sp);
const float grad_tsp = (1 - tsp*tsp) * grad_sp;
const float grad = inp * grad_tsp + tsp;
delta[i] *= grad;

//float x = activation_input[i];
//float d = 2 * expf(x) + expf(2 * x) + 2;
//float w = 4 * (x + 1) + 4 * expf(2 * x) + expf(3 * x) + expf(x)*(4 * x + 6);
//float derivative = expf(x) * w / (d * d);
//delta[i] *= derivative;
}
}