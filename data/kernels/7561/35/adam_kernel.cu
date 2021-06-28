#include "includes.h"
__global__ void adam_kernel(int N, float *x, float *m, float *v, float B1, float B2, float rate, float eps, int t)
{
int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if (index >= N) return;

x[index] = x[index] - (rate * sqrtf(1.F-powf(B2, t)) / (1.F-powf(B1, t)) * m[index] / (sqrtf(v[index]) + eps));
//if(index == 0) printf("%f %f %f %f\n", m[index], v[index], (rate * sqrtf(1.F-powf(B2, t)) / (1.F-powf(B1, t)) * m[index] / (sqrt(v[index]) + eps)));
}