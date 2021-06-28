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