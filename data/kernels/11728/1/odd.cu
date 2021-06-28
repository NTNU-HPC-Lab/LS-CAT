#include "includes.h"



__global__ void odd(int *darr, int n) {
int k = threadIdx.x;
int t;
k = k * 2 + 1;
if (k <= n - 2) {
if (darr[k] > darr[k + 1]) {
t = darr[k];
darr[k] = darr[k + 1];
darr[k + 1] = t;
}
}

}