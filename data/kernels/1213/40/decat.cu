#include "includes.h"
__global__ void decat(float* output1, float* output2, float* output3, float* output4, size_t num1, size_t num2, size_t num3, size_t num4, size_t maxNum, float* input, const int numPerBatch)
{
size_t i = blockDim.x * blockIdx.x + threadIdx.x;


for(;i < maxNum; i += size_t(blockDim.x * gridDim.x)){
size_t batchIdx = i / numPerBatch; // which batch this thread is working in
const int batchOffset = i - batchIdx * numPerBatch; // offset of current thread in current batch

if(batchOffset < num1){  // first output
output1[batchOffset + batchIdx * num1] = input[i];
}
else if(batchOffset < (num1 + num2)){  // second output
output2[(batchOffset - num1) + batchIdx * num2] = input[i];
}
else if(batchOffset < (num1 + num2 + num3)){  // third input
output3[(batchOffset - (num1 + num2)) + batchIdx * num3] = input[i];
}
else{  // fourth input
output4[(batchOffset - (num1 + num2 + num3)) + batchIdx * num4] = input[i];
}
}
}