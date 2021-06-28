#include "includes.h"
__global__ void matMul(float *A, int l_A, int c_A, float *B, int l_B, int c_B, float *C, int l_C, int c_C)
{
float resultat = 0.0;
int ligne = blockDim.x * blockIdx.x + threadIdx.x;
int colonne = blockDim.y * blockIdx.y + threadIdx.y;

if(ligne > l_A || colonne > c_B)
{
printf("ERREUR - Soit ligne > m soit colonne > m\n");
return;
}

for(int i = 0; i < c_A; i++)
resultat += A[ligne * c_A + i] * B[i * c_B + colonne];

C[ligne * c_C + colonne] = resultat;
}