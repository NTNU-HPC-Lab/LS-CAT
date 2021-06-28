#include "includes.h"
__global__ void bitonic_sort(int* arrayIn, int* arrayOut, int arrayLen, int chunkSize){
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < arrayLen) {
int myValue = arrayIn[idx];
int chunkStart = (idx / chunkSize) * chunkSize;
int chunkMid = chunkStart + (chunkSize / 2);
int partnerIndex = chunkSize - (idx - chunkStart) - 1 + chunkStart;
if (partnerIndex < arrayLen) {
int partnerValue = arrayIn[partnerIndex];
int min = (myValue <= partnerValue) ? myValue:partnerValue;
int max = (myValue > partnerValue) ? myValue:partnerValue;
myValue = (idx < chunkMid) ? min:max;
}
arrayOut[idx] = myValue;
}
}