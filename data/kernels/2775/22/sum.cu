#include "includes.h"
__global__ void sum(const float *input, float *output, int numElements) {
float val = 0.f;
for (int i = 0; i < numElements; ++i) {
val += input[i];
}
*output = val;
}