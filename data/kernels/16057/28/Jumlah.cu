#include "includes.h"
__global__ void Jumlah(float *sumMatrix,float *mulMatrix){
int Index = blockIdx.x * blockDim.x + threadIdx.x;
if(Index<1) printf("%f",mulMatrix[0]);
atomicAdd(&sumMatrix[0],mulMatrix[Index]);

}