#include "includes.h"
__global__ void FullToCOO(int numElem, float* H_vals, double* hamilValues, int dim)
{

int i = threadIdx.x + blockDim.x*blockIdx.x;

if (i < numElem)
{

hamilValues[i] = H_vals[i];


}
}