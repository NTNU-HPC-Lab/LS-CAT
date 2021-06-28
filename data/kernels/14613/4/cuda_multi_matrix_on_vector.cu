#include "includes.h"
__global__ void cuda_multi_matrix_on_vector(int *matrix, int *vector, int *new_vector, int numElements){
__shared__ int cache[threadsPerBlock];
const int idx = blockDim.x*blockIdx.x + threadIdx.x;//глобальный индекс
const int tIdx = threadIdx.x;//индекс нити
const int k = (numElements - 1 + threadsPerBlock) / threadsPerBlock;//всего кол-во блоков

for (int i = 0; i < k; i++){//в блок влезает threadsPerBlock нитей. Чтобы посчитать всю строку на нужно читать кусок вектора k раз
if (tIdx+threadsPerBlock*i < numElements){//если индекс нити плюс потоковое смещение меньше n то копируем в память shared
cache[tIdx] = vector[tIdx + threadsPerBlock * i];
}
__syncthreads();

int min = numElements - i*threadsPerBlock;//определяем хвост
if (min > threadsPerBlock)min = threadsPerBlock;//если хвост слишком длинный то берём по нитям
if (idx < numElements){
for (int j= 0; j < min; j++){
new_vector[idx] += cache[j]*matrix[(i*threadsPerBlock + j)*numElements + idx];//каждая нить считает свой вектор умножая кусок вектора на сообверствующий кусок матрицы
}
}
__syncthreads();
}
}