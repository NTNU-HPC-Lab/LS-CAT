#include "includes.h"



#define Columnas 10
#define Filas 10
cudaError_t addWithCuda(int* c, const int* a, unsigned int size);


__device__ unsigned int computeOutputEdge(int mask[][3], int vecinos[][3], int rows, int cols) {

float result = 1;
int sum = 0;

for (int i = 0; i < rows; i++) {
for (int j = 0; j < cols; j++) {
float mul = mask[i][j] * vecinos[i][j];
sum = sum + mul;
}
}
result = abs(sum);
return (int)result;
}
__global__ void bordes(int* val2, int* val1, int m, int n)
{

int column = threadIdx.x + blockDim.x * blockIdx.x;
int row = threadIdx.y + blockDim.y * blockIdx.y;

int myEdge[3][3] = { {0,1,0},{1,-4,1},{0,1,0} };
//int filas = (sizeof(myMask)/sizeof(myMask[0]));

if (row < m && column < n) {

int thread_id1 = (row - 1) * n + (column - 1);
int thread_id2 = (row - 1) * n + (column);
int thread_id3 = (row - 1) * n + (column + 1);

int thread_id4 = (row)* n + (column - 1);

int thread_id5 = (row)* n + (column);

int thread_id6 = (row)* n + (column + 1);

int thread_id7 = (row + 1) * n + (column - 1);
int thread_id8 = (row + 1) * n + (column);
int thread_id9 = (row + 1) * n + (column + 1);

//int my_val = val1[thread_id5];

//printf("row: %d, \tcol: %d, \tvalor: %d\n", row, column, my_val);

val2[thread_id5] = val1[thread_id5];

if ((row > 0 && row < (m - 1)) && (column > 0 && column < (n - 1)))
{
int my_val0 = val1[thread_id1];
int my_val2 = val1[thread_id2];
int my_val3 = val1[thread_id3];
int my_val4 = val1[thread_id4];
int my_val5 = val1[thread_id5]; //doubly-subscripted access
int my_val6 = val1[thread_id6];
int my_val7 = val1[thread_id7];
int my_val8 = val1[thread_id8];
int my_val9 = val1[thread_id9];
//printf("row: %d, col: %d, value: %d\n", row, column, my_val);

int myMask2[3][3] = { {(my_val0),(my_val2),(my_val3)},
{(my_val4),(my_val5),(my_val6)},
{(my_val7),(my_val8),(my_val9)} };

unsigned int output = computeOutputEdge(myEdge, myMask2, 3, 3);
//printf("row: %d,\t col: %d,\t Valor Original: %d,\t Nuevo Valor: %d\n", row, column, my_val5,output);
//printf("Salida: %d \n", output);
//printf("Entro\n");
val2[thread_id5] = output;
}
}
}