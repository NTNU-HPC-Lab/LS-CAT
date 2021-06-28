#include "includes.h"
__global__ void twiddleRealKernel(float *wr, float *w, int N)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;
int i = 0, index;

if (idx < N) {
if (idx == 0) {
for (i = 0; i < N; i++)
wr[idx * N + i] = 1;
} else {
wr[idx * N + 0] = 1;
for (i = 1; i < N; i++) {
index = (idx * i) % N;
wr[idx * N + i] = w[index * 2];
}
}
}
}