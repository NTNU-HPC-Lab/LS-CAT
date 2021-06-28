#include "includes.h"
__global__ void calc(float *d_D, int n, int k){ //kernel (4  cells for every thread)
__shared__ float s_d[4*3*256]; //Shared table within a block
int i = blockIdx.x * blockDim.x + threadIdx.x; //Calculation of i and j
int j = blockIdx.y * blockDim.y + threadIdx.y;
int b_index = 4 * 3 * (threadIdx.x + blockDim.x*threadIdx.y); //Calculation of initial index of thread in the shared table within the block
int istep = blockDim.x*gridDim.x, jstep = blockDim.y*gridDim.y;
int l, m , v=0;
for (l = 0; l<2; l++){
for (m = 0; m<2; m++){ //Pass values from device table to shared block table for every one of the 4 cells
s_d[b_index + 3 * v] = d_D[(i+l*istep)+(j+m*jstep)*n];
s_d[b_index + (3 * v + 1)] = d_D[(i + l*istep) + k*n];
s_d[b_index + (3 * v + 2)] = d_D[k + (j + m*jstep)*n];
v++;
}
}
for (v = 0; v<4; v++){ //Calculate the new cell values (4 for every thread)
if (s_d[b_index + 3 * v] > s_d[b_index + (3 * v + 1)] + s_d[b_index + (3 * v + 2)]) s_d[b_index + 3 * v] = s_d[b_index + (3 * v + 1)] + s_d[b_index + (3 * v + 2)];
}
v = 0;
for (l = 0; l<2; l++){ //Pass the new values to the device table
for (m = 0; m<2; m++){
d_D[(i+l*istep)+(j+m*jstep)*n] = s_d[b_index + 3 * v];
v++;
}
}
}