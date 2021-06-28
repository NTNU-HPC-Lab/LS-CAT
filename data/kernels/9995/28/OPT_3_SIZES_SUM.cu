#include "includes.h"
__global__ void OPT_3_SIZES_SUM(int* lcmsizes, int n) {

for(int i = 0; i < n; i++)
lcmsizes[i+1] += lcmsizes[i];
}