#include "includes.h"
__global__ void testMemset(float* array, float value, int N){
int i = threadindex;
if(i < N){
array[i] = value;
}
}