#include "includes.h"
__global__ void SumaColMatrizKernel (int M, int N, float *Md, float *Nd){
// Pvalue es usado para el valor intermedio
__shared__ float Nds[DIMBLOCKX];
float Pvalue = 0;
int columna = blockIdx.y*(N/gridDim.x)+threadIdx.x;
int pasos = M/blockDim.x ;
int posIni = columna * M + threadIdx.x * pasos;
for (int k = 0; k < pasos; ++k) {
Pvalue = Pvalue + Md[posIni + k];
}
Nds[threadIdx.x] = Pvalue;
__syncthreads();
if (threadIdx.x == 0 ){
for (int i = 1; i < blockDim.x; ++i) {
Nds[0] = Nds[0]+Nds[i];
}

Nd[columna*gridDim.x+blockIdx.x] = Nds[0];

}
}