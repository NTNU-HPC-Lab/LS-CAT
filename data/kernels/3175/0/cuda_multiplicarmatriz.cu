#include "includes.h"
/*
** Projeto de Algoritmos Paralelos
** Multiplicação de Matrizes
*/


#define TAM_BLOCO 16


// Função para rodar na CPU
// Computa R = M * N
//   aM é a altura de M
//   lM é a largura de M
//   lN é a largura de N
__global__ void cuda_multiplicarmatriz(float* M, float* N, float* R, int tamM, int tamN) {

//índice do bloco
int bx = blockIdx.x;
int by = blockIdx.y;

// índice da thread
int tx = threadIdx.x;
int ty = threadIdx.y;

// índice da primeira submatriz de M processado pelo bloco
int mComeco = tamM * TAM_BLOCO * by;

// índice da última submatriz de M processada pelo bloco
int mFim   = mComeco + tamM - 1;

// Tamanho do passo utilizado para interar através das submatrizes de M
int mPasso  = TAM_BLOCO;

// Índice da primeira submatriz de N processada pelo bloco
int nComeco = TAM_BLOCO * bx;

// Tamanho do passo utilizado para interar através das submatrizes de N
int nPasso  = TAM_BLOCO * tamN;

// O elemento computado pela thread
float rRes = 0;

// Varre por todas as submatrizes de M e N requeridas
// para computar o bloco de submatriz
for (int m = mComeco, n = nComeco; m <= mFim; m += mPasso, n += nPasso) {

// Memoria compartilhada para a submatriz de M
__shared__ float Msub[TAM_BLOCO][TAM_BLOCO];

// Memoria compartilhada para a submatriz de N
__shared__ float Nsub[TAM_BLOCO][TAM_BLOCO];

// Carrega as matrizes da memória global para a memória
// compartilhada. Cada thread carreg um elemento de cada
// matriz
Msub[ty][tx] = M[m + tamM * ty + tx];
Nsub[ty][tx] = N[n + tamN * ty + tx];

// Sincroniza para garantir que todas as matrizes foram
// carregadas
__syncthreads();

// Multiplica as duas matrizes.
// Cada thread computa um elemento
// do bloco da submatriz
for (int i = 0; i < TAM_BLOCO; ++i)
rRes += Msub[ty][i] * Nsub[i][tx];

// Sincroniza para grantir que a computação de multiplicação
// está feita antes de carregar duas novas submatrizes de
// M e N na próxima interação
__syncthreads();
}
// Esscre o bloco da sumatriz na memória global
// Cada thread escreve  um único elemento
int r = tamN * TAM_BLOCO * by + TAM_BLOCO * bx;
R[r + tamN * ty + tx] = rRes;
}