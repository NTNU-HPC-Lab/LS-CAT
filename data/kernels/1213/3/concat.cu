#include "includes.h"
__global__ void concat(float* output, float* input1, float* input2, float* input3, float* input4, const int size, const int in_channel1, const int in_channel2, const int in_channel3, const int in_channel4)
{
const int pos = blockIdx.x * blockDim.x + threadIdx.x;
const int out_channel = in_channel1 + in_channel2 + in_channel3 + in_channel4;  // # of channel for output
const int N = size * size;  // total elements per channel

if(pos < N){
for(int n = 0; n < out_channel; n++){
const int row = pos / size;
const int col = pos % size;
if(n < in_channel1){  // first input
output[(n * size + col) * size + row] = input1[(n * size + col) * size + row];
}
else if(n < in_channel1 + in_channel2){  // second input
output[(n * size + col) * size + row] = input2[((n - in_channel1) * size + col) * size + row];
}
else if(n < in_channel1 + in_channel2 + in_channel3){  // third input
output[(n * size + col) * size + row] = input3[((n - in_channel1 - in_channel2) * size + col) * size + row];
}
else{  // last input
output[(n * size + col) * size + row] = input4[((n - in_channel1 - in_channel2 - in_channel3) * size + col) * size + row];
}
}
}
}