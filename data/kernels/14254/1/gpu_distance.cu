#include "includes.h"
__global__ void gpu_distance(int* data, float* distance, int* point, int n, int dim) {
int i = blockIdx.x * blockDim.x + threadIdx.x;

if (i >= n)
return;

float d = 0;

for(int j = 0; j<dim; j++)
d += abs(data[i*dim + j] - point[j]);

distance[i] = d;
}