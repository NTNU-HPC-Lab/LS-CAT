#include "includes.h"
/*
Now we make the matrix much bigger
g++ -pg seq_matrix_big_mul.c -o seq_matrix_big_mul
*/

#define N_THREADS 32

int num_rows_A = 2000; int num_rows_B = 2000; int num_rows_C = 2000;
int num_cols_A = 2000; int num_cols_B = 600; int num_cols_C = 600;
//int num_rows_A = 64; int num_rows_B = 64; int num_rows_C = 64;
//int num_cols_A = 64; int num_cols_B = 64; int num_cols_C = 64;

// I'm forcing a malloc because I want to add the malloc time on the game
float *A = (float*) malloc(sizeof(float) * num_rows_A * num_cols_A);
float *B = (float*) malloc(sizeof(float) * num_rows_B * num_cols_B);
float *C = (float*) malloc(sizeof(float) * num_rows_C * num_cols_C);
float *C_ref = (float*) malloc(sizeof(float) * num_rows_C * num_cols_C);


__global__ void matrix_2d_mul_float_gpu(float *A, float *B, float *C, int num_rows_A, int num_cols_A, int num_cols_B) {
// Same code for all 2d kernel
int i = blockIdx.y * blockDim.y + threadIdx.y;
int k = blockIdx.x * blockDim.x + threadIdx.x;
if (i > num_rows_A || k > num_cols_B) return;

float sum = 0;

for (int j=0; j<num_cols_A; j++){
// A[i][j] == A[i*num_cols_A+j]
// B[j][k] == B[j*num_cols_B+k]
//sum += A[i][j]*B[j][k];
sum += A[i*num_cols_A+j]*B[j*num_cols_B+k];
}

C[i*num_cols_B+k]=sum;
}