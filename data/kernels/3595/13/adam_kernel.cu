#include "includes.h"
__global__ void adam_kernel(int N, float *x, float *m, float *v, float B1, float B2, float rate, float eps, int t)
{
int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if (index >= N) return;

float mhat = m[index] / (1.f - powf(B1, t));
float vhat = v[index] / (1.f - powf(B2, t));

x[index] = x[index] + rate * mhat / (sqrtf(vhat) + eps);
}