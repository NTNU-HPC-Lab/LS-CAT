#include "includes.h"
__global__ void reverseArray(int *A, int *B) {
int threadID = threadIdx.x;
int start = (threadID * ArraySize) / 256;
int end = ( ( (threadID + 1 ) * ArraySize) / 256) - 1;
while(end > 0)
{
B[end] = A[start];
end--;
start++;
}
}