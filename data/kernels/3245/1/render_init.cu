#include "includes.h"

//Limited version of checkCudaErrors from helper_cuda.h

#define checkCudaErrors(val) check_errors( (val), #val, __FILE__, __LINE__ )

__global__ void render_init(int width, int length, curandState *rand_state) {
int i = threadIdx.x + blockIdx.x * blockDim.x;
int j = threadIdx.y + blockIdx.y * blockDim.y;
if ((i >= width) || (j >= length)) {
return;
}
int index = j * width+ i;
//Each thread gets same seed, a different sequence number, no offset
curand_init(1984, index, 0, &rand_state[index]);
}