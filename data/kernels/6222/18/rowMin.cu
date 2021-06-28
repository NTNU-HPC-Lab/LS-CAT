#include "includes.h"
__global__ void rowMin(float* input, int* output, size_t rowS, size_t rowNum){
size_t id = blockIdx.x*blockDim.x + threadIdx.x;

if(id < rowNum){
float temp[MAX_K/2][2];
size_t inId = id * rowS;

for(int i = 0; i< rowS;i++){
temp[i][0] = input[inId + i];
temp[i][1] = (float)i;
}

for(int i = 0; i< rowS; i++){
float best = temp[i][0];
int bestInd = i;
for(int j = i; j < rowS; j++){
if(temp[j][0] > best){
best = temp[j][0];
bestInd = j;
}
}
float iVal = temp[i][0];
float iInd = temp[i][1];
temp[i][0] = temp[bestInd][0];
temp[i][1] = temp[bestInd][1];
temp[bestInd][0] = iVal;
temp[bestInd][1] = iInd;
}

for(int i = 0; i< rowS; i++){
output[inId+i] = (int)temp[i][1];
}
}
}