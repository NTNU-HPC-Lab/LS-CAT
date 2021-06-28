#include "includes.h"
__global__ void naiveKernel(int N, float *input, float *output){
float res = 0.;
for(int i=0;i<N;++i) res += input[i];
*output = res/N;
}