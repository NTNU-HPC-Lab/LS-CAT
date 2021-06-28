#include "includes.h"


__global__ void vector_add_cu(float *out, float *a, float *b, int n){
for(int i = 0; i < n; i++){
out[i] = a[i] + b[i];
}
}