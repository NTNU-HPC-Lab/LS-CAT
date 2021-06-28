#include "includes.h"
__global__ void concat(float* input1, float* input2, float* input3, float* input4, size_t num1, size_t num2, size_t num3, size_t num4, size_t maxNum, float* output, const int numPerBatch)
{
size_t i = blockDim.x * blockIdx.x + threadIdx.x;

for(;i < maxNum; i += size_t(blockDim.x * gridDim.x)){
size_t batchIdx = i / numPerBatch; // which batch this thread is working in
const int batchOffset = i - batchIdx * numPerBatch; // offset of current thread in current batch

if(batchOffset < num1){  // first input
output[i] = input1[batchOffset + batchIdx * num1];
}
else if(batchOffset < (num1 + num2)){  // second input
output[i] = input2[(batchOffset - num1) + batchIdx * num2];
}
else if(batchOffset < (num1 + num2 + num3)){  // third input
output[i] = input3[(batchOffset - (num1 + num2)) + batchIdx * num3];
}
else{  // fourth input
output[i] = input4[(batchOffset - (num1 + num2 + num3)) + batchIdx * num4];
}
}
}