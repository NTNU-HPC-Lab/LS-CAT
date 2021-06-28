#include "includes.h"
__global__ void cudaComputeSignature(double* hyperplanes, double* v, int* dimensions, bool* sig, long* hyperp_length) {
long tid = threadIdx.x + blockDim.x * blockIdx.x;

if (tid < *hyperp_length) {
int d_dimensions = *dimensions;
long pos = tid * d_dimensions;
double sum = 0.0;

for (int i = 0; i < d_dimensions; i++)
sum += hyperplanes[i+pos] * v[i];
sig[tid] = (sum>=0);
}
}