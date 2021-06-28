#include "includes.h"
__global__ void RevealNumber(int* number, unsigned int number_size)
{
printf("CudaDevice()::RevealNumber()\n");
unsigned int idx = blockDim.x * gridDim.x + threadIdx.x;
if (idx < number_size)
{
printf("Here comes: %i", number[idx]);
}
}