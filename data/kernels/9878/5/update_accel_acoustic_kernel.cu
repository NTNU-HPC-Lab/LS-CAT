#include "includes.h"
__global__ void update_accel_acoustic_kernel(float * accel, const int size, const float * rmass){
int id;
id = threadIdx.x + (blockIdx.x) * (blockDim.x) + (blockIdx.y) * ((gridDim.x) * (blockDim.x));
if (id < size) {
accel[id] = (accel[id]) * (rmass[id]);
}
}