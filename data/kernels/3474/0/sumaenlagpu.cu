#include "includes.h"


#define N (4096*4096)
#define HILOS_POR_BLOQUE 512


__global__ void sumaenlagpu(int *a, int *b, int *c, int n){
int index = threadIdx.x + blockIdx.x*blockDim.x;
if (index < n){
c[index] = a[index] + b[index];
}
}