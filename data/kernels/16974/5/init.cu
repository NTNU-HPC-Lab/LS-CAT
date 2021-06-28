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
__global__ void init(unsigned int seed, curandState_t* states) {
int idx = threadIdx.x+blockDim.x*blockIdx.x;

curand_init(seed, idx, 0,  &states[idx]);
}