#include "includes.h"
__global__ void mm_tiled(float *dA, float *dB, float *dC, int DIM, int N, int GPUN) {
int it, jt, kt, i, j, k;
__shared__ float sA[32][32], sB[32][32];

// (it, jt) => the first element of a specific tile
it = blockIdx.y * 32;
jt = blockIdx.x * 32;

// (i, j) => specific element
i = it + threadIdx.y;
j = jt + threadIdx.x;

if (i*DIM+j <= GPUN) {
float sum = 0.0f;
// per tile loop
for (kt = 0; kt < DIM; kt += 32) {
// copy to shared memory
sA[threadIdx.y][threadIdx.x] = dA[(it+threadIdx.y)*DIM + kt + threadIdx.x];
sB[threadIdx.y][threadIdx.x] = dB[(kt+threadIdx.y)*DIM + jt + threadIdx.x];
__syncthreads();

// two 32x32 small shared (dB[it + 0:31][kt + 0:31], dC[kt+0:31][jt + 0:31]) at this point
for (k = kt; k < kt+32; k++) {
sum += sA[i-it][k-kt] * sB[k-kt][j-jt];
}

__syncthreads();
}
dC[i*DIM+j] = sum;
}
}