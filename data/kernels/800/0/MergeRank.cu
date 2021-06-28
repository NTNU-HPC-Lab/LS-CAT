#include "includes.h"
__global__ void MergeRank(float * d_input, float * d_output)
{
int indexA = blockIdx.x * blockDim.x + threadIdx.x;
int indexB = indexA + 2048;
float temp1 = d_input[indexA];
float temp2 = d_input[indexB];
int indexAB = 2048;
while (d_input[indexAB] < temp1) {
indexAB++;
}
int indexBA = 0;
while (d_input[indexBA] < temp2) {
indexBA++;
}
__syncthreads();
d_output[indexA + indexAB + 1] = temp1;
d_output[indexB + indexBA + 1] = temp2;

}