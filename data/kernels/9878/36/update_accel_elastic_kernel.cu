#include "includes.h"
__global__ void update_accel_elastic_kernel(float * accel, const float * veloc, const int size, const float two_omega_earth, const float * rmassx, const float * rmassy, const float * rmassz){
int id;
id = threadIdx.x + (blockIdx.x) * (blockDim.x) + (blockIdx.y) * ((gridDim.x) * (blockDim.x));
if (id < size) {
accel[(id) * (3)] = (accel[(id) * (3)]) * (rmassx[id]) + (two_omega_earth) * (veloc[(id) * (3) + 1]);
accel[(id) * (3) + 1] = (accel[(id) * (3) + 1]) * (rmassy[id]) - ((two_omega_earth) * (veloc[(id) * (3)]));
accel[(id) * (3) + 2] = (accel[(id) * (3) + 2]) * (rmassz[id]);
}
}