#include "includes.h"

//this function is a kernel
//__global__ is a C construct called declaration specifier and that’s how CUDA knows that this is not CPU code but a kernel
//threadIdx: CUDA has a built in variable called threadIdx which tells each thread its index within a block. Its a C construct
//with 3 members “x”, “y” and “z” and the struct is called “dim3"

__global__ void cube(float * d_out, float * d_in){
int idx = threadIdx.x; //
float f = d_in[idx];
d_out[idx] = f*f*f;
}