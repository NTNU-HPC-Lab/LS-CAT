#include "includes.h"
__global__ void sumArrayOnGPU(float *A, float *B, float *C){
// スレッドIDを割り当てる
int i = threadIdx.x;
C[i] = A[i] + B[i];
}