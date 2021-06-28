#include "includes.h"
__global__ void computePointGradients1(float *x, float *y, float *z, int points_num, int *valid_points, int valid_points_num, double *dj_ang, double *pg24, double *pg05, double *pg15, double *pg25)
{
int id = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;
__shared__ double j_ang[12];


if (threadIdx.x < 12) {
j_ang[threadIdx.x] = dj_ang[threadIdx.x + 12];
}

__syncthreads();

for (int i = id; i < valid_points_num; i += stride) {
int pid = valid_points[i];

//Orignal coordinates
double o_x = static_cast<double>(x[pid]);
double o_y = static_cast<double>(y[pid]);
double o_z = static_cast<double>(z[pid]);

//Compute point derivatives

pg24[i] = o_x * j_ang[0] + o_y * j_ang[1] + o_z * j_ang[2];
pg05[i] = o_x * j_ang[3] + o_y * j_ang[4] + o_z * j_ang[5];
pg15[i] = o_x * j_ang[6] + o_y * j_ang[7] + o_z * j_ang[8];
pg25[i] = o_x * j_ang[9] + o_y * j_ang[10] + o_z * j_ang[11];
}
}