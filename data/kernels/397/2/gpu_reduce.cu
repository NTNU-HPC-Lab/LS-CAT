#include "includes.h"
/**
* Computación Paralela (curso 1516)
*
* Alberto Gil
* Guillermo Cebrian
*/


// Includes generales


// Include para las utilidades de computación paralela

/**
* Estructura antena
*/
typedef struct {
int y;
int x;
} Antena;

/**
* Macro para acceder a las posiciones del mapa
*/

#define m(y,x) mapa[ (y * cols) + x ]

/**
* Definimos el tamaño de bloque
*/
#define TAMBLOCK 128

/* Funcion que inicializa la matriz al valor maximo */


/* Actualiza el mapa despues de colocar una antena*/




/**
* Función de ayuda para imprimir el mapa
*/

__global__ void gpu_reduce(int *c, int size)
{
/*Identificaciones necesarios*/
int IDX_Thread = threadIdx.x;
int IDY_Thread = threadIdx.y;
int IDX_block =	blockIdx.x;
int IDY_block =	blockIdx.y;
int shapeGrid_X = gridDim.x;
int threads_per_block =	blockDim.x * blockDim.y;
int position = threads_per_block * ((IDY_block * shapeGrid_X)+IDX_block)+((IDY_Thread*blockDim.x)+IDX_Thread);


if(position<size){
if(size%2 != 0)
{
if(c[position]<c[size-1])
{
c[position]=c[size-1];
}
}else{

if(c[position]<c[position+size/2])
{
c[position]=c[position+size/2];
}
}
}
}