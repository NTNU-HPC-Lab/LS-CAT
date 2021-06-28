#include "includes.h"
__global__ void compute_infection_prob_kernel(double alpha, double beta, int *infectious_rat_count, int *exposed_rat_count, int *rat_count, double *infection_prob_result, int width, int height) {
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int nid = y * width + x;
if(x < width && y < height) {
if(rat_count[nid] == 0) {
infection_prob_result[nid] = 0.0;
} else {
double density_of_exposed = (double)(exposed_rat_count[nid]) / (double)(rat_count[nid]);
double density_of_infectious = (double)(infectious_rat_count[nid]) / (double)(rat_count[nid]);
infection_prob_result[nid] = alpha * density_of_infectious + beta * density_of_exposed;
}
}
}