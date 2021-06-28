#include "includes.h"
__global__ void cudaDoEigen(double* m, int rows, int columns)
{
//    Eigen::Matrix stuff = m;
//    printf("CUDA testing\n");
printf("CUDA ptr: %p\n", m);
printf("CUDA value: %lf\n", m[0]);
printf("CUDA value: %lf\n", m[1]);
printf("CUDA value: %lf\n", m[2]);
printf("CUDA value: %lf\n", m[3]);

}