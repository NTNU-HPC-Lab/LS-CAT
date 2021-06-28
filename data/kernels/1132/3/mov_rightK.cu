#include "includes.h"


using namespace std;

void showMatriz(int *matriz, int anchura, int altura);
void generateSeeds(int *matriz, int ancho, int alto, int cantidad, char modo);
void gestionSemillas(int *matriz, int ancho, int numeroSemillas, int alto, char modo);
int checkFull(int *matriz, int tamano);
bool checkMove(int *matriz, int ancho, int alto);
void guardar(int vidas, int *tablero, int altura, int anchura, char dificultad);
int* cargar();
int* MostrarEspecificaciones();

cudaError_t cudaStatus;

/*	add_up
*	Función del kernel para sumar hacia arriba todos los números que sean iguales.
*/
__device__ void stack_right(int *matriz, int anchura, int altura, int x, int y)
{

for (int i = anchura - 1; i > 0; i--) //realizaremos el desplazamiento celda a celda una altura-1 veces para gestionar la posibilidad del ultimo poniendose el primero de la lista
{
if ((y != anchura-1) && (matriz[x*anchura + y] != 0) && matriz[x*anchura + (y + 1)] == 0) //Si la celda pertenece a la primera fila, es 0 o su superior no es 0, no hace nada
{
matriz[x*anchura + (y + 1)] = matriz[x*anchura + y]; //Si lo es, desplazamos la celda
matriz[x*anchura + y] = 0;
}
__syncthreads(); //utilizamos una sincronizacion para que estos pasos sean realizados a la vez por los hilos del bloque
}
}
__device__ void add_right(int *matriz, int x, int y, int altura, int anchura)
{
if (y != anchura-1 && y < anchura) //Los primeros hilos de la derecha no deben realizar ninguna operacion pues serán modificados por los demas
{
if (matriz[x*anchura + y] != 0) //Si es distinto de 0, gestiona su posible suma o desplazamiento
{
if (matriz[x*anchura + y] == matriz[x*anchura + (y + 1)]) //Si es igual a su superior, se procede a comprobar el numero de celdas con el mismo numero que hay en esa columna

{
int iguales = 0;
iguales++;
for (int i = 1; y + i < anchura; i++)
{
if (matriz[x*anchura + y] == matriz[x*anchura + (y + i)])
{
iguales++;
}
else {
break;
}
}
if (iguales % 2 == 0) //Si el numero es par, se suman, si no, ese numero será mezclado con otro y no estará disponible
{
matriz[x*anchura + (y + 1)] = matriz[x*anchura + (y + 1)] * 2;
matriz[x*anchura + y] = 0;
}
}
else if (matriz[x*anchura + (y + 1)] == 0) // Se comprueba que otros hilos hayan dejado 0 en sus operaciones para desplazarse
{
matriz[x*anchura + (y + 1)] = matriz[x*anchura + y];
matriz[x*anchura + y] = 0;
}
}
}
}
__global__ void mov_rightK(int *matriz, int anchura, int altura)
{
int x = threadIdx.x;
int y = threadIdx.y;

stack_right(matriz, anchura, altura, x, y); //Realizamos las llamadas de la siguiente manera para gestionar el movimiento:
add_right(matriz, x, y, altura, anchura); //2 2 0 4   -> 4 4 0 0
__syncthreads();
stack_right(matriz, anchura, altura, x, y);
}