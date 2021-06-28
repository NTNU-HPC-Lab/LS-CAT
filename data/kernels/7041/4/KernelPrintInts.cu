#include "includes.h"
__global__ void KernelPrintInts(const int* p, int len) {
for (int i = 0; i < len; ++i) {
printf("%d\n", p[i]);
}
}