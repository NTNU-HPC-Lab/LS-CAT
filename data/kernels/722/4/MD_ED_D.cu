#include "includes.h"
__global__ void MD_ED_D(float *S, float *T, int trainSize, int window_size, int dimensions, float *data_out, int task, int gm) {

long long int i, j, p;
float sumErr = 0, dd = 0;
int idx = blockIdx.x * blockDim.x + threadIdx.x;

if (gm == 0) {
extern __shared__ float T2[];

int t, offset;
if (task == 0) {

offset = window_size;

int wind = dimensions * window_size;
t = idx * wind;
if ((idx * wind) + wind >
trainSize * wind) // CHANGE FORMULA 120=train_size
return;

if (threadIdx.x == 0) {
for (i = 0; i < dimensions; i++)
for (j = 0; j < window_size; j++)
T2[window_size * i + j] = T[window_size * i + j];
}

__syncthreads();
} else {

// in this case 'trainSize' is the number of subsequence to search 'nss',
// that is, the length of dataset to perform on
offset = trainSize;

t = idx;
if ((idx + window_size) > trainSize)
return;

if (threadIdx.x == 0) {
for (i = 0; i < dimensions; i++)
for (j = 0; j < window_size; j++)
T2[window_size * i + j] = T[window_size * i + j];
}
__syncthreads();
}

for (j = 0; j < window_size; j++) {
dd = 0;
for (p = 0; p < dimensions; p++)
dd += (S[(t + p * offset) + j] - T2[(p * window_size) + j]) *
(S[(t + p * offset) + j] - T2[(p * window_size) + j]);

sumErr += dd;
}
data_out[idx] = sqrt(sumErr);
} else {

int t, offset;
if (task == 0) {

offset = window_size;

int wind = dimensions * window_size;
t = idx * wind;
if ((idx * wind) + wind > trainSize * wind)
return;
} else {

// in this case 'trainSize' is the number of subsequence to search 'nss',
// that is, the length of dataset to perform on
offset = trainSize;

t = idx;
if ((idx + window_size) > trainSize)
return;
}

for (j = 0; j < window_size; j++) {
dd = 0;
for (p = 0; p < dimensions; p++)
dd += (S[(t + p * offset) + j] - T[(p * window_size) + j]) *
(S[(t + p * offset) + j] - T[(p * window_size) + j]);

sumErr += dd;
}
data_out[idx] = sqrt(sumErr);
}
}