#include "includes.h"
using namespace std;

__global__ void setValue(float *data, int idx, float value) {
if(threadIdx.x == 0) {
data[idx] = value;
}
}