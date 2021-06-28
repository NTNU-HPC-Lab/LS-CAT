#include "includes.h"
__global__ void SSTTest(float* V, int* addr, int N) {
int i = threadIdx.x;
if (i < N) {
int return_val = 0;
float element = V[i];
asm("/*");
asm("CPTX_BEGIN");
asm("sst.sstarr.f32 %0, [%1], %2, %3;" : "=r"(return_val) : "l"(&V[0]), "r"(i), "f"(element)); // perform SST instruction
asm("CPTX_END");
asm("*/");
if (return_val != 0) *addr = (int)(return_val - (intptr_t)&V[0])/4; // last thread stores the result
}
}