#include "includes.h"
__global__ void getqss(double *IN, double *qss, int N, int t) {
int I, i, j;
i = 10; j = 10;
I = j*N + i;
qss[t] = IN[I];
}