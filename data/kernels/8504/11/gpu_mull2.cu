#include "includes.h"
__global__ void gpu_mull2(float* a, float* b, float* c, int n, int m,int p)
{
int i = blockIdx.x * 32 + threadIdx.x;
int j = blockIdx.y;

float sum = 0.0f;
for (int k = 0; k < p; ++k) {
sum += b[i + n * k] * c[k + p * j];

}
a[i + n * j] = sum;
}