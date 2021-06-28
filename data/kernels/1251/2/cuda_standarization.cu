#include "includes.h"
__global__ void cuda_standarization(float *data, int rows, int columns) {
int total_threads_count = blockDim.x * gridDim.x;
int tid = threadIdx.x + blockIdx.x * blockDim.x;
float var, ave, amo;

for (int i = tid+1; i < columns; i=i+total_threads_count) {
amo = 0, var = 0;
for (int j = 0; j < rows; ++j) {
amo = amo + *(data + (j * columns) + i);
}
ave  = amo / float(rows);

for (int j = 0; j < rows; ++j) {
float factor = *(data + (j * columns) + i) - ave;
var = var + (factor * factor);
}

if (var == 0) {
for (int j = 0; j < rows; j++) {
*(data + (j * columns) + i) = *(data + (j * columns) + i) / 255.;
}
continue;
}

float sd_reciprocal = 1./sqrt(var);

for (int j = 0; j < rows; j++) {
*(data + (j * columns) + i) = (*(data + (j * columns) + i) - ave) * sd_reciprocal;
}
}
}