#include "includes.h"
__global__ void calcularBloques(int *matriz, int *u, int *resultado, int num_bloques, int nc, int m ){
int index1 = threadIdx.x + blockIdx.x*blockDim.x; // 0 - 1
int index2 = threadIdx.y + blockIdx.y*blockDim.y; // 0 - 1
int suma = 0;

for(int i=0 ; i < num_bloques ; i++){
suma = 0;
for(int l=0 ; l < nc ; l++){
suma += matriz[l+index1*nc] * u[index2+m*(l+i*nc)];
}
resultado[index2 + m*(index1+i*nc)] = suma;
}
}