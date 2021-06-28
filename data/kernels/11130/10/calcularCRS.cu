#include "includes.h"
__global__ void calcularCRS(int *val, int *col_ind, int *row_ptr, int *u, int *resultado, int l ){
int i = threadIdx.x + blockIdx.x*blockDim.x; // 0 - 9
int j = threadIdx.y + blockIdx.y*blockDim.y; // 0 - 9
int suma = 0;

for(int k = row_ptr[i]-1; k < row_ptr[i+1]-1; k++){
suma += val[k] * u[j + ( (col_ind[k]-1) * l) ];
}
resultado[j+i*l] = suma;
}