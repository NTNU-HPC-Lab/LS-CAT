#include "includes.h"
__global__ void update_veloc_elastic_kernel(float * veloc, const float * accel, const int size, const float deltatover2){
int id;
id = threadIdx.x + (blockIdx.x) * (blockDim.x) + (blockIdx.y) * ((gridDim.x) * (blockDim.x));
if (id < size) {
veloc[id] = veloc[id] + (deltatover2) * (accel[id]);
veloc[size + id] = veloc[size + id] + (deltatover2) * (accel[size + id]);
veloc[size + size + id] = veloc[size + size + id] + (deltatover2) * (accel[size + size + id]);
}
}