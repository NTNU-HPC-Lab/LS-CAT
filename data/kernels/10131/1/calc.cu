#include "includes.h"
__global__ void calc(float *d_D, int n, int k){ //kernel
__shared__ float s_d[3*256]; //shared in block table of floats (size 3*number threads/block)
int i = blockIdx.x * blockDim.x + threadIdx.x;  //We find i & j in the Grid of threads
int j = blockIdx.y * blockDim.y + threadIdx.y;
int b_index = 3 * (threadIdx.x + blockDim.x*threadIdx.y); //Calculation of initial index in shared table s_d
s_d[b_index] = d_D[i + j*n];  //Pass values from device table to shared
s_d[b_index + 1] = d_D[i + k*n];
s_d[b_index + 2] = d_D[k + j*n];
if (s_d[b_index] > s_d[b_index + 1] + s_d[b_index + 2]) s_d[b_index] = s_d[b_index + 1] + s_d[b_index + 2]; //Calculation of new distance value
d_D[i + j*n] = s_d[b_index]; //Pass the values back to the table s_d
}