#include "includes.h"
__global__ void rMD_ED_D(float *S, float *T, int window_size, int dimensions, float *data_out, int trainSize, int gm) {

long long int i, j, p;
float sumErr = 0, dd = 0;
int idx = blockIdx.x * blockDim.x + threadIdx.x;

if (gm == 0) {

extern __shared__ float T2[];

// offset training set
int s = dimensions * 2 * window_size * (idx / window_size);
int t = s + idx % window_size;

if (idx >= (trainSize * window_size)) //
return;

if (threadIdx.x == 0) {
for (i = 0; i < dimensions; i++)
for (j = 0; j < window_size; j++)
T2[window_size * i + j] = T[window_size * i + j];
}
__syncthreads();

for (j = 0; j < window_size; j++) {
dd = 0;
for (p = 0; p < dimensions; p++)
dd += (S[(t + p * 2 * window_size) + j] - T2[(p * window_size) + j]) *
(S[(t + p * 2 * window_size) + j] - T2[(p * window_size) + j]);

sumErr += dd;
}
data_out[idx] = sqrt(sumErr);
} else {

int s = dimensions * 2 * window_size * (idx / window_size);
int t = s + idx % window_size;

if (idx >= (trainSize * window_size))
return;

for (j = 0; j < window_size; j++) {
dd = 0;
for (p = 0; p < dimensions; p++)
dd += (S[(t + p * 2 * window_size) + j] - T[(p * window_size) + j]) *
(S[(t + p * 2 * window_size) + j] - T[(p * window_size) + j]);

sumErr += dd;
}
data_out[idx] = sqrt(sumErr);
}
}