#include "includes.h"
extern "C" {
}
__global__ void reduce_sum_final(const float* x, float* y, unsigned int len) {
*y = 0;
for(int i = 0; i < len; ++i) {
*y += x[i];
}
}