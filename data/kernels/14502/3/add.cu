#include "includes.h"
__global__ void add(int *a, int *b, int *c, int *d, int *e, int *f) {
*c = *a + *b;
*d = *a - *b;
*e = *a * *b;
*f = *a / *b;
}