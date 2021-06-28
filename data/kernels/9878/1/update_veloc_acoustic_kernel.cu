#include "includes.h"
__global__ void update_veloc_acoustic_kernel(float * veloc, const float * accel, const int size, const float deltatover2){
int id;
id = threadIdx.x + (blockIdx.x) * (blockDim.x) + (blockIdx.y) * ((gridDim.x) * (blockDim.x));
if (id < size) {
veloc[id] = veloc[id] + (deltatover2) * (accel[id]);
}
}