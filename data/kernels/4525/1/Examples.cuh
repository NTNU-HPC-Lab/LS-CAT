#include <crt/host_defines.h>

__global__ void hello_cuda();
__global__ void print_thread_id();
__global__ void print_thread_variables();
__global__ void print_unique_thread_id_1D();

__global__ void print_unique_thread_id_3D(int * data);

