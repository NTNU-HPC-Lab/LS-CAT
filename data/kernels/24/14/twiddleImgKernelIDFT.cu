#include "includes.h"
__global__ void twiddleImgKernelIDFT(float *wi, float *w, int N)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;
int i, index;

if (idx < N) {
if (idx == 0) {
for (i = 0; i < N; i++)
wi[idx * N + i] = 0;
} else {
wi[idx * N + 0] = 0;
for (i = 1; i < N; i++) {
index = (idx * i) % N;
wi[idx * N + i] = w[index * 2 + 1];
}
}
}
}