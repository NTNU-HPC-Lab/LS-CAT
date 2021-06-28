#include "includes.h"
__global__ void CombineScreen(float* d_postEdge1, float* d_postEdge2, float* d_postGradient1, float* d_postGradient2, float* d_postGradient3, float* d_postSobel3LR, float* d_postSobel3UD, float* d_postSmooth31, float* d_output){
int id = threadIdx.x + blockDim.x * blockIdx.x;

for (int i = 0; i < 73; ++i){
d_output[i + id * 73 + 73 * 73 * 0] =     d_postEdge1[id * 73 + i];
d_output[i + id * 73 + 73 * 73 * 1] =     d_postEdge2[id * 73 + i];
d_output[i + id * 73 + 73 * 73 * 2] = d_postGradient1[id * 73 + i];
d_output[i + id * 73 + 73 * 73 * 3] = d_postGradient2[id * 73 + i];
d_output[i + id * 73 + 73 * 73 * 4] = d_postGradient3[id * 73 + i];
d_output[i + id * 73 + 73 * 73 * 5] =  d_postSobel3LR[id * 73 + i];
d_output[i + id * 73 + 73 * 73 * 6] =  d_postSobel3UD[id * 73 + i];
d_output[i + id * 73 + 73 * 73 * 7] =  d_postSmooth31[id * 73 + i];
}
}