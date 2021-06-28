#include "includes.h"










__global__ void smoothGray (unsigned char *imagem, unsigned char *saida, unsigned int cols, unsigned int linhas)
{
unsigned int indice = (blockIdx.y * blockDim.x * 65536) + (blockIdx.x * 1024) + threadIdx.x; // calcula o indice do vetor com base nas dimensões de bloco e indice da thread
if(indice >= cols*linhas)
return;
//indices para o campo da imagem que participará do smooth
int i_begin = (indice/(int)cols) - 2, i_end = (indice/(int)cols)+2;
int j_begin = (indice%(int)cols) - 2, j_end = (indice%(int)cols)+2;
if(i_begin<0) i_begin = 0;
if(j_begin<0) j_begin = 0;
if(i_end>=cols) i_end = cols-1;
if(j_end>=cols) j_end = cols-1;

//calcula o smooth no ponto de indice da thread
int media = 0;
int qtd = 0;
for (int i = i_begin; i<=  i_end; ++i)
{
for(int j = j_begin; j<= j_end; ++j)
{
media += imagem[(i*cols)+j];
qtd++;
}
}
saida[indice] = (unsigned char)(media/qtd);
}