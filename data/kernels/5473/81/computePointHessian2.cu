#include "includes.h"
__global__ void computePointHessian2(float *x, float *y, float *z, int points_num, int *valid_points, int valid_points_num, double *dh_ang, double *ph155, double *ph165, double *ph175)
{
int id = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;
__shared__ double h_ang[9];

if (threadIdx.x < 9) {
h_ang[threadIdx.x] = dh_ang[36 + threadIdx.x];
}

__syncthreads();

for (int i = id; i < valid_points_num; i += stride) {
int pid = valid_points[i];

//Orignal coordinates
double o_x = static_cast<double>(x[pid]);
double o_y = static_cast<double>(y[pid]);
double o_z = static_cast<double>(z[pid]);

ph155[i] = o_x * h_ang[0] + o_y * h_ang[1] + o_z * h_ang[2];
ph165[i] = o_x * h_ang[3] + o_y * h_ang[4] + o_z * h_ang[5];
ph175[i] = o_x * h_ang[6] + o_y * h_ang[7] + o_z * h_ang[8];

}
}