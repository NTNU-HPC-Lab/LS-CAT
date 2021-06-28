#include "includes.h"

#define N 1024 //wielkoæ obliczanych wektorów
#define imin(a, b) (a<b?a:b)
const int threadsPerBlock = 256; //iloæ w¹tków na k¹zdy blok
const int blocksPerGrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);//iloæ wykorzystywanych bloków





__global__ void multiplyMatrix(float *a, float *b, float *c) {
__shared__ float cache[threadsPerBlock]; //Zmienna dzielona ze wszystkimi w¹tkami w tym bloku. Nie dzieli siê z innymi blokami!

int tid = threadIdx.x + blockIdx.x * blockDim.x; //id w¹tku który to wykonuje, id w¹tku + id bloku * pojemnoæ bolku
int cacheIndex = threadIdx.x; //id cache, które jest takie samo jak id obecnego w¹tku

float temp = 0;
while (tid < N) {
temp = a[tid] * b[tid]; //zapis mno¿enia w zmiennej
tid += blockDim.x * gridDim.x; //przesuwanie o iloæ wszystkich w¹tków w ca³ej siatce, nie trzeba ogarniaæ na czwórkê
}

cache[cacheIndex] = temp; //przypisanie wyniku mno¿enia do wspó³dzielonej tablicy cache

__syncthreads(); //czekanie a¿ wszystkie w¹tki dotr¹ to tego miejsca

//tu trochê w powalony sposób sumuj¹ siê wszystkie wyniki
int i = blockDim.x / 2;
while (i != 0) {
if (cacheIndex < i) {
cache[cacheIndex] += cache[cacheIndex + i];
}

__syncthreads();
i /= 2;
}
//przypisanie sumy wszystkich wyników mno¿enia do tablicy c
if (cacheIndex == 0)
c[blockIdx.x] = cache[0]; //jako, ¿e cache nie jest wspó³dzielony pomiêdzy blokami to wyników bêdzie tyle ile by³o wykorzystanych bloków, póniej to siê sumuje na cpu

}