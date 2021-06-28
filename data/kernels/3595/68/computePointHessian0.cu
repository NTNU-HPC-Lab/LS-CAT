#include "includes.h"
__global__ void computePointHessian0(float *x, float *y, float *z, int points_num, int *valid_points, int valid_points_num, double *dh_ang, double *ph93, double *ph103, double *ph113, double *ph123, double *ph94, double *ph133, double *ph104, double *ph143, double *ph114, double *ph153, double *ph95, double *ph163, double *ph105, double *ph173, double *ph115)
{
int id = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;
__shared__ double h_ang[18];

if (threadIdx.x < 18) {
h_ang[threadIdx.x] = dh_ang[threadIdx.x];
}

__syncthreads();

for (int i = id; i < valid_points_num; i += stride) {
int pid = valid_points[i];

//Orignal coordinates
double o_x = static_cast<double>(x[pid]);
double o_y = static_cast<double>(y[pid]);
double o_z = static_cast<double>(z[pid]);


ph93[i] = 0;
ph103[i] = o_x * h_ang[0] + o_y * h_ang[1] + o_z * h_ang[2];
ph113[i] = o_x * h_ang[3] + o_y * h_ang[4] + o_z * h_ang[5];

ph123[i] = ph94[i] = 0;
ph133[i] = ph104[i] = o_x * h_ang[6] + o_y * h_ang[7] + o_z * h_ang[8];
ph143[i] = ph114[i] = o_x * h_ang[9] + o_y * h_ang[10] + o_z * h_ang[11];

ph153[i] = ph95[i] = 0;
ph163[i] = ph105[i] = o_x * h_ang[12] + o_y * h_ang[13] + o_z * h_ang[14];
ph173[i] = ph115[i] = o_x * h_ang[15] + o_y * h_ang[16] + o_z * h_ang[17];

}
}