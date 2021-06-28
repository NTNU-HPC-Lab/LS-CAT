#include "includes.h"
__global__ void computePointHessian1(float *x, float *y, float *z, int points_num, int *valid_points, int valid_points_num, double *dh_ang, double *ph124, double *ph134, double *ph144, double *ph154, double *ph125, double *ph164, double *ph135, double *ph174, double *ph145)
{
int id = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;
__shared__ double h_ang[18];

if (threadIdx.x < 18) {
h_ang[threadIdx.x] = dh_ang[18 + threadIdx.x];
}

__syncthreads();

for (int i = id; i < valid_points_num; i += stride) {
int pid = valid_points[i];

//Orignal coordinates
double o_x = static_cast<double>(x[pid]);
double o_y = static_cast<double>(y[pid]);
double o_z = static_cast<double>(z[pid]);

ph124[i] = o_x * h_ang[0] + o_y * h_ang[1] + o_z * h_ang[2];
ph134[i] = o_x * h_ang[3] + o_y * h_ang[4] + o_z * h_ang[5];
ph144[i] = o_x * h_ang[6] + o_y * h_ang[7] + o_z * h_ang[8];

ph154[i] = ph125[i] = o_x * h_ang[9] + o_y * h_ang[10] + o_z * h_ang[11];
ph164[i] = ph135[i] = o_x * h_ang[12] + o_y * h_ang[13] + o_z * h_ang[14];
ph174[i] = ph145[i] = o_x * h_ang[15] + o_y * h_ang[16] + o_z * h_ang[17];
}
}