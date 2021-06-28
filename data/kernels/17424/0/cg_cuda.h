#ifndef CG_CUDA_H_
#define CG_CUDA_H_

#include <stdio.h> 
#include <stdlib.h> 
#include <string.h>
#include <math.h>
#include <cblas.h>
#include <cuda.h>
#include "second.h"	

#define MAX_THREADS_PER_BLOCK 256

double cgsolver(double *A_cuda, double *X_cuda, double *Y_cuda, int *m_locals_cuda, int *A_all_pos_cuda, double *b, double *x, int n, int nthreads);
void init_cuda(double **A_cuda, double **X_cuda, double **Y_cuda, int **m_locals_cuda, int **A_all_pos_cuda, double *A, int *m_locals, int *A_all_pos, int n, int nthreads);
void finalize_cuda(double *A_cuda, double *X_cuda, double *Y_cuda, int *m_locals_cuda, int *A_all_pos_cuda);
void mvm(double *A_cuda, double *X_cuda, double *Y_cuda, int *m_locals_cuda, int *A_all_pos_cuda, double *X, double *Y, int n, int nthreads);

#endif /*CG_CUDA_H_*/


