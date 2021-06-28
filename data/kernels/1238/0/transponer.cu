#include "includes.h"
__global__ void transponer(float* entrada, float* salida, int ANCHO){
int tx = blockIdx.x*blockDim.x + threadIdx.x;
int ty = blockIdx.y*blockDim.y + threadIdx.y;
salida[tx*ANCHO + ty] = entrada[ty*ANCHO + tx];
}