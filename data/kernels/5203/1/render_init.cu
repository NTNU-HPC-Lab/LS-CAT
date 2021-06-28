#include "includes.h"

using namespace std;

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
int i = threadIdx.x + blockIdx.x * blockDim.x;
int j = threadIdx.y + blockIdx.y * blockDim.y;
if((i >= max_x) || (j >= max_y)) return;
int pixel_index = j*max_x + i;
//Each thread gets same seed, a different sequence number, no offset
curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}