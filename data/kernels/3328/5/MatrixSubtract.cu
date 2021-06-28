#include "includes.h"
__global__ void MatrixSubtract(const float* A_elements, const float* B_elements,  float* C_elements, const int size)
{
int thread = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;


for(int i = thread; i < size; i += stride)
//Modifying array of elements of Matrix C
C_elements[i] = A_elements[i] - B_elements[i];
}