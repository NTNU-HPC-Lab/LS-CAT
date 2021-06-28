#include "includes.h"
__global__ void populate_reverse_pad(const double *Q, double *Q_reverse_pad, const double *mean, const int window_size, const int size)
{
int tid = blockIdx.x * blockDim.x + threadIdx.x;
double mu = *mean;
if(tid < window_size) {
Q_reverse_pad[tid] = Q[window_size - 1 - tid] - mu;
}else if(tid < size){
Q_reverse_pad[tid] = 0;
}
}