#include "includes.h"
__global__ void AddIntsCUDA(int* a, int* b) {
for (int i = 0; i < 1000005; i++) {
a[0] += b[0];
}
}