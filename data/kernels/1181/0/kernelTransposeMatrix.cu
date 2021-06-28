#include "includes.h"
/*
* PARA CORRERLO:
*   $ export LD_LIBRARY_PATH=/usr/local/cuda/lib
*   $ export PATH=$PATH:/usr/local/cuda/bin
*   $ nvcc -o matrixTrans matrixTrans.cu -O2 -lc -lm
*   $ ./matrixTrans n
*/

/*
* UNSIGNED INT --> Tipo de dato para enteros, números sin punto decimal.
*                  Los enteros sin signo pueden ser tan grandes como 65535
*                  y tan pequeños como 0.
*                  Son almacenados como 16 bits de información.
*
* SIZE_T --> is an unsigned integer type guaranteed to support the longest
*            object for the platform you use. It is also the result of the
*            sizeof operator.sizeof returns the size of the type in bytes.
*            So in your context of question in both cases you pass a
*            size_t to malloc.
*/

#define NUMBER_THREADS 32

float elapsed_time_ms;
int gpudev = 1;

char *dev_mat_in, *dev_mat_out;

//---------------------------------------------------------------------------

__global__ void kernelTransposeMatrix(const char *mat_in, char *mat_out, unsigned int rows, unsigned int cols){
unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;

if (idx < cols && idy < rows) {
unsigned int pos = idy * cols + idx;
unsigned int trans_pos = idx * rows + idy;
mat_out[trans_pos] = mat_in[pos];
}
}