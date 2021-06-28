#include "includes.h"
__global__ void compute_absv(const unsigned int nSpheres, const float* velX, const float* velY, const float* velZ, float* d_absv) {
unsigned int my_sphere = blockIdx.x * blockDim.x + threadIdx.x;
if (my_sphere < nSpheres) {
float v[3] = {velX[my_sphere], velY[my_sphere], velZ[my_sphere]};
d_absv[my_sphere] = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}
}