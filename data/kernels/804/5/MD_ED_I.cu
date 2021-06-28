#include "includes.h"
__global__ void MD_ED_I(float *S, float *T, int trainSize, int window_size, int dimensions, float *data_out, int task, int gm) {

int idx, offset_x;
float sumErr = 0;
long long int i, j;


if(gm == 0){

extern __shared__ float sh_mem[];

float *T2 = (float *)sh_mem;
float *DTW_single_dim =
(float *)&sh_mem[dimensions * window_size]; // offset on the shared memory
// for the segment T2

if (task == 0) {
idx = threadIdx.x * dimensions + threadIdx.y;
offset_x = ((blockDim.x * blockDim.y * window_size) * blockIdx.x) +
idx * window_size;

if (((blockDim.x * blockDim.y * blockIdx.x) + idx) >=
trainSize * dimensions) // 120=train_size
return;

} else { // SUBSEQ_SEARCH

idx = threadIdx.x * dimensions + threadIdx.y;
offset_x =
(blockDim.x * blockIdx.x) +
((threadIdx.y * trainSize) +
threadIdx.x); // use blockIdx and other measure to set well the offset

if ((idx + window_size) > trainSize)
return;
}

if (idx == 0) {
for (i = 0; i < dimensions; i++)
for (j = 0; j < window_size; j++)
*(T2 + (window_size * i + j)) = T[window_size * i + j];
}
__syncthreads();

for (j = 0; j < window_size; j++)
sumErr += (S[offset_x + j] - T2[window_size * threadIdx.y + j]) *
(S[offset_x + j] - T2[window_size * threadIdx.y + j]);

DTW_single_dim[idx] = sqrt(sumErr);

__syncthreads();

if (idx == 0) {
for (i = 0; i < blockDim.x; i++) {
data_out[(blockIdx.x * blockDim.x) + i] = 0.0;
for (j = 0; j < blockDim.y; j++) {
data_out[(blockIdx.x * blockDim.x) + i] +=
DTW_single_dim[i * dimensions + j]; // rivedere formula!
}
}
}

}
else {

extern __shared__ float DTW_single_dim[];

if (task == 0) {
idx = threadIdx.x * dimensions + threadIdx.y;
offset_x = ((blockDim.x * blockDim.y * window_size) * blockIdx.x) +
idx * window_size;

if (((blockDim.x * blockDim.y * blockIdx.x) + idx) >=
trainSize * dimensions) // 120=train_size
return;

} else { // SUBSEQ_SEARCH

idx = threadIdx.x * dimensions + threadIdx.y;
offset_x =
(blockDim.x * blockIdx.x) +
((threadIdx.y * trainSize) +
threadIdx.x); // use blockIdx and other measure to set well the offset

if ((idx + window_size) > trainSize)
return;
}

// if (idx == 0) {
//   for (i = 0; i < dimensions; i++)
//     for (j = 0; j < window_size; j++)
//       *(T2 + (window_size * i + j)) = T[window_size * i + j];
// }
// __syncthreads();

for (j = 0; j < window_size; j++)
sumErr += (S[offset_x + j] - T[window_size * threadIdx.y + j]) *
(S[offset_x + j] - T[window_size * threadIdx.y + j]);

DTW_single_dim[idx] = sqrt(sumErr);

__syncthreads();

if (idx == 0) {
for (i = 0; i < blockDim.x; i++) {
data_out[(blockIdx.x * blockDim.x) + i] = 0.0;
for (j = 0; j < blockDim.y; j++) {
data_out[(blockIdx.x * blockDim.x) + i] +=
DTW_single_dim[i * dimensions + j]; // rivedere formula!
}
}
}

}

}