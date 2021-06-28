#include "includes.h"



struct MPIGlobalState {
// The CUDA device to run on, or -1 for CPU-only.
int device = -1;

// A CUDA stream (if device >= 0) initialized on the device
cudaStream_t stream;

// Whether the global state (and MPI) has been initialized.
bool initialized = false;
};

// MPI relies on global state for most of its internal operations, so we cannot
// design a library that avoids global state. Instead, we centralize it in this
// single global struct.
static MPIGlobalState global_state;

// Initialize the library, including MPI and if necessary the CUDA device.
// If device == -1, no GPU is used; otherwise, the device specifies which CUDA
// device should be used. All data passed to other functions must be on that device.
//
// An exception is thrown if MPI or CUDA cannot be initialized.
__global__ void kernel_add(const float* x, const float* y, const int N, float* out) {
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
out[i] = x[i] + y[i];
}
}