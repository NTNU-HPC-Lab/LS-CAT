#include "includes.h"
__global__ void setMultiLHS ( double* dsMulti, double* dlMulti, double* diagMulti, double* duMulti, double* dwMulti,  double a, double b, double c, double d, double e,  int nx, int batchCount )
{
// Matrix index
int globalIdx = blockDim.x * blockIdx.x + threadIdx.x;
int globalIdy = blockDim.y * blockIdx.y + threadIdx.y;

// Index access
int index = globalIdy * batchCount + globalIdx;

if (globalIdx < batchCount && globalIdy < nx)
{

dsMulti[index] = a;

dlMulti[index] = b;

diagMulti[index] = c;

duMulti[index] = d;

dwMulti[index] = e;

}
}