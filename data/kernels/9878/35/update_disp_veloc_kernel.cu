#include "includes.h"
__global__ void update_disp_veloc_kernel(float * displ, float * veloc, float * accel, const int size, const float deltat, const float deltatsqover2, const float deltatover2){
int id;
id = threadIdx.x + (blockIdx.x) * (blockDim.x) + (blockIdx.y) * ((gridDim.x) * (blockDim.x));
if (id < size) {
displ[id] = displ[id] + (deltat) * (veloc[id]) + (deltatsqover2) * (accel[id]);
veloc[id] = veloc[id] + (deltatover2) * (accel[id]);
accel[id] = 0.0f;
}
}