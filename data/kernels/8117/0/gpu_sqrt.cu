#include "includes.h"

long N = 6400000000;
int doPrint = 0;

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// HELPER CODE TO INITIALIZE, PRINT AND TIME
struct timeval start, end;
__global__ void gpu_sqrt(float* a, long N) {
long element = blockIdx.x*blockDim.x + threadIdx.x;
if (element < N) a[element] = sqrt(a[element]);
}