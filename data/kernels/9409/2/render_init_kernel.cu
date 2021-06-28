#include "includes.h"

/*
* Cuda kernels that does the heavy work
*/




////////////////////////////////////////////////////////////////////////////////
// declaration, forward
extern "C" void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line);


__global__ void render_init_kernel(int max_x, int max_y, curandState *rand_state) {
int i = threadIdx.x + blockIdx.x * blockDim.x;
int j = threadIdx.y + blockIdx.y * blockDim.y;
if ((i >= max_x) || (j >= max_y)) return;
int pixel_index = j * max_x + i;
//Each thread gets same seed, a different sequence number, no offset
curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}