#include "includes.h"


__global__ void scale_centroids(int d, int k, int* counts, double* centroids) {
int global_id_x = threadIdx.x + blockIdx.x * blockDim.x;
int global_id_y = threadIdx.y + blockIdx.y * blockDim.y;
if ((global_id_x < d) && (global_id_y < k)) {
int count = counts[global_id_y];
//To avoid introducing divide by zero errors
//If a centroid has no weight, we'll do no normalization
//This will keep its coordinates defined.
if (count < 1) {
count = 1;
}
double scale = 1.0/double(count);
centroids[global_id_x + d * global_id_y] *= scale;
}
}