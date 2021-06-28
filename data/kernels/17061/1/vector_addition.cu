#include "includes.h"
__global__  void vector_addition (int *a, int *b, int *c, int n) {
for (int i=0; i<n; i++) {
c[i] = a[i] + b[i];
}
}