#include "includes.h"
using namespace std;



__global__ void variance(int* n, double *x, double *mean)
{
int index = threadIdx.x;
int stride = blockDim.x;

for (int i = index; i < *n; i+= stride) {
x[i] = (x[i] - *mean) ;
x[i] = x[i] * x[i];
}
}