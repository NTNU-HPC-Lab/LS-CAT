#include "includes.h"
__global__ void kernel_sub(char* newB, char* first, char* second, int size_biggest, int diff, int * size_newB) {
int tmp = 0;
int i = threadIdx.x;
#if __CUDA_ARCH__>=200
//printf("#threadIdx.x = %d\n", threadIdx.x);
#endif
if (i == 0) return;

//for (int i = size_biggest - 1; i >= 0; i--) {
if (i - 1 - diff >= 0 && (second[i - 1 - diff] != '+' && second[i - 1 - diff] != '-')) {
tmp = first[i - 1] - second[i-1-diff];
} else if (first[i - 1] != '+' && first[i - 1] != '-') {
tmp = first[i - 1];
}

if (tmp < 0) {
// warning 10 - tmp ?
newB[i - 1]--;
tmp += 10;
}
if (i != 0)
newB[i] += tmp;
//}
}