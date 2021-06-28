#include "includes.h"
__global__ void findMaxIndMultipleDetector(float *input, int* maxInd, int size)
{
int maxIndex = 0;
int count = 1;

for (int i = 1; i < size; i++){
if (input[maxIndex] < input[i]){
maxIndex = i;
count = 1;
}
else if (input[maxIndex] == input[i]){
count++;
}
}
if(count>1)
maxInd[0] = -1;
else
maxInd[0] = maxIndex;
}