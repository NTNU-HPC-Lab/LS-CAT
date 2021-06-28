#include "includes.h"

#define INF 2147483647

extern "C" {





}
__global__ void init(int * tab, int len) {
for(int i = threadIdx.x + len*blockIdx.x; i < len*blockIdx.x + len; i += 1024) {
tab[i] = INF;
}
}