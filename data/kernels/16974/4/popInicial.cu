#include "includes.h"

//Bibliotecas Basicas

//Biblioteca Thrust


//Biblioteca cuRAND


//PARAMETROS GLOBAIS
const int QUANT_PAIS_AVALIA = 4;
int POP_TAM = 200;
int N_CIDADES = 20;
int BLOCKSIZE = 1024;
int TOTALTHREADS = 2048;
int N_GERA = 100;
const int MUT = 10;
const int MAX = 19;
const int MIN = 0;
const int ELITE = 2;

/*
* Busca por erros nos processos da gpu
*/
__global__ void popInicial(unsigned int n,unsigned int np,int* v, int* genes, int* ale)
{
int index = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;
for (int i=index; i<n; i+=stride)
{
for(int j=0; j<np; j++)
{
int p = (ale[i*np+j]<j)?j:ale[i*np+j];
v[i*np+j] = genes[i*np+p];
int aux = genes[i*np+j];
genes[i*np +j] = genes[i*np+p];
genes[i*np+p]=aux;
}
}
}