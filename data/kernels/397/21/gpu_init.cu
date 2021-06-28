#include "includes.h"
__global__ void gpu_init(int *mapad, int max, int size){

/*Identificaciones necesarios*/
int IDX_Thread = threadIdx.x;	/*Identificacion del hilo en la dimension*/
int IDY_Thread = threadIdx.y;	/*Identificacion del hilo en la dimension y*/
int IDX_block =	blockIdx.x;	/*Identificacion del bloque en la dimension x*/
int IDY_block = blockIdx.y;	/*Identificacion del bloque en la dimension y */
int shapeGrid_X = gridDim.x; 	/*Numeros del bloques en la dimension */

int threads_per_block =	blockDim.x * blockDim.y; /* Numero de hilos por bloque (1 dimension) */

/*Formula para calcular la posicion*/	//Posicion del vector dependiendo del hilo y del bloque
int position = threads_per_block * ((IDY_block * shapeGrid_X)+IDX_block)+((IDY_Thread*blockDim.x)+IDX_Thread);

//inicializamos
if(position<size) mapad[position] = max;
}