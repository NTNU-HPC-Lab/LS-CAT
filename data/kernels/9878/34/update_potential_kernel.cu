#include "includes.h"
__global__ void update_potential_kernel(float * potential_acoustic, float * potential_dot_acoustic, float * potential_dot_dot_acoustic, const int size, const float deltat, const float deltatsqover2, const float deltatover2){
int id;
id = threadIdx.x + (blockIdx.x) * (blockDim.x) + (blockIdx.y) * ((gridDim.x) * (blockDim.x));
if (id < size) {
potential_acoustic[id] = potential_acoustic[id] + (deltat) * (potential_dot_acoustic[id]) + (deltatsqover2) * (potential_dot_dot_acoustic[id]);
potential_dot_acoustic[id] = potential_dot_acoustic[id] + (deltatover2) * (potential_dot_dot_acoustic[id]);
potential_dot_dot_acoustic[id] = 0.0f;
}
}