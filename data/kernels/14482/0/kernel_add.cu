#include "includes.h"
__device__ inline void charAtomicAdd(char *address, char value) {
int oldval, newval, readback;

oldval = *address;
newval = oldval + value;
while ((readback=atomicCAS((int *)address, oldval, newval)) != oldval) {
oldval = readback;
newval = oldval + value;
}
}
__global__ void kernel_add(char* newB, char* first, char* second, int size_biggest, int diff, int * size_newB) {
int tmp = 0;
int i = threadIdx.x;
#if __CUDA_ARCH__>=200
//printf("#threadIdx.x = %d\n", threadIdx.x);
#endif
if (i == 0) return;

//for (int i = size_biggest - 1; i >= 0; i--) {
if (i - 1 - diff >= 0 && (second[i - 1 - diff] != '+' && second[i - 1 - diff] != '-')) {
tmp = second[i - 1 - diff] + first[i - 1];
} else if (first[i - 1] != '+' && first[i - 1] != '-') {
tmp = first[i - 1];
}

if (tmp >= 10) {
//charAtomicAdd(&newB[i], 1);
newB[i - 1]++;
tmp = tmp % 10;
}
if (i != 0)
newB[i] += tmp;
//}
}