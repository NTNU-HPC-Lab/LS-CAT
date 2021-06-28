#include "includes.h"



__global__ void even(int *darr, int n) {
int k = threadIdx.x;
int t;
k = k * 2;
if (k <= n - 2) {
if (darr[k] > darr[k + 1]) {
t = darr[k];
darr[k] = darr[k + 1];
darr[k + 1] = t;
}
}
}