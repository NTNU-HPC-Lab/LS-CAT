#include "includes.h"
__global__ void matrixTriUpper(float *a, int m, int n) {
//setting matricies to their upper bound
for(int i = 0; i < m; ++i) {
for(int j = 0; j < n; ++j) {
if(i>j)
a[i*n + j] = 0;
a[i*n + j] = a[i*n + j];
}
}
}