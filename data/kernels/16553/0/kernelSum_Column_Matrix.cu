#include "includes.h"

using namespace std;
const int DIMBLOCKX=32;
//DEVICE



//HOST
__global__ void kernelSum_Column_Matrix(float* matrix, float* array, int tam){
__shared__ float shareMatrix[DIMBLOCKX];

float value=0;
int col=blockIdx.x;
int step= tam/blockDim.x;
int posIni= col*tam+threadIdx.x*step;
for(int i=0;i<step;i++){
value=value+matrix[posIni+i];
}

shareMatrix[threadIdx.x]=value;
__syncthreads();

if(threadIdx.x==0){
for(int j=1;j<blockDim.x;j++){
shareMatrix[0]=shareMatrix[0]+shareMatrix[j];
}
array[blockIdx.x]=shareMatrix[0];
}
}