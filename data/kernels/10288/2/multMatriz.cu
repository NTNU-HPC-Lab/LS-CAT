#include "includes.h"
__global__ void multMatriz(float *da, float *db, float *dc, int num){
float sum=0;
int j = threadIdx.x + blockIdx.x * blockDim.x;
int i = threadIdx.y + blockIdx.y * blockDim.y;
while(j<num){
while(i<num){
for (unsigned int k = 0; k<num; k++)
sum += da[i * num + k] * db[k * num + j];
dc[i*num + j] = (float) sum;
i += gridDim.y * blockDim.y;
}
j+=gridDim.x * blockDim.x;
i = threadIdx.y + blockIdx.y * blockDim.y;
}

}