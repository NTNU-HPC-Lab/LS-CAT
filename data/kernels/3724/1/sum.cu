#include "includes.h"
/*
152096 - William Matheus
Friendly Numbers
Programacao Paralela e Distribuida
CUDA - 2019/2 - UPF
Programa 2 - Kernel
*/



__global__ void sum(long int* device_num, long int* device_den, long int* device_vet, int size, int x)
{
int i = blockIdx.x * blockDim.x + threadIdx.x + x;
int j;

if (i < size) {
for (j = i + 1; j < size; j++) {
if ((device_num[i] == device_num[j]) && (device_den[i] == device_den[j]))
device_vet[i]++;
}
}
}