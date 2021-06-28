#include "includes.h"
__global__ void decryptKernel(char* deviceDataIn, char* deviceDataOut, int n, char *key, int keySize) {
unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
if (index < n)
deviceDataOut[index] = deviceDataIn[index] - key[index % keySize];
}