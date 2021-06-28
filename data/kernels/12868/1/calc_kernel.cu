#include "includes.h"

#define TOLERANCE 0.00001
#define TRUE 1
#define FALSE 0

long usecs();
void initialize(double **A, int rows, int cols);
int calc_serial(double **A, int rows, int cols, int iters, double tolerance);
int calc_serial_v1(double **A, int rows, int cols, int iters, double tolerance);
int calc_omp(double **A, int rows, int cols, int iters, double tolerance, int num_threads);
int calc_gpu(double **A, int rows, int cols, int iters, double tolerance);
double verify(double **A, double **B, int rows, int cols);


__global__ void calc_kernel(double* w, double* r, int rows, int cols, double tolerance) {
int row = blockIdx.x;
int col = threadIdx.x;
int idx = row*blockDim.x + col;
if (row < rows && row > 0 && col < cols) {
w[idx] = 0.2*(r[idx+1] + r[idx - 1] + r[(row-1)*blockDim.x + col] + r[(row+1)*blockDim.x + col]);
}
}