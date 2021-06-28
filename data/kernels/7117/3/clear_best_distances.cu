#include "includes.h"

__global__ void clear_best_distances(int *best_distances, int rays_number) {
int i = blockDim.x * blockIdx.x + threadIdx.x;
if (i >= rays_number)
return;

best_distances[i] = INT32_MAX;
}