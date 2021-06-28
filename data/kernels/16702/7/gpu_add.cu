#include "includes.h"
__global__ void gpu_add(int* big_set_numbers, const int big_set_count, int* tiny_set_numbers,const int tiny_set_count)		// __global__ prefix i vscc tarafindan anlasilmaz ve bu fonksiyonu nvcc compile edecektir
{
extern __shared__ int tiny_shared[];		// shared memory uzerinde depolanacak array.. toplam shared memory alani block sayisina bolunerek, her dilim bir block icin tahsis edilir..

int tidX = threadIdx.x;

if (tidX < tiny_set_count)					// threadid, tiny_set_count tan kucuk oldudgu surece extern yapili tiny_shared i doldur..
{
tiny_shared[tidX] = tiny_set_numbers[tidX];
}
// blockDim sayisi, tiny_set_count tan fazla olabilir ve fazlalik thread ler sonraki satirlara gecebilir..
__syncthreads();			// tum thread lerin bu satira gelmesi beklenir.. Yani __syncthreads() bir bariyer gorevi gorur
// tum thread ler __syncthreads() e gelmeden, hic bir thread sonraki satira gecemez..
// tum thread lerin shread memory uzerinde tiny_shared i doldurmasindan sonra shared memory alanina, tiny_shared icin erisim yap

int id = blockIdx.x * blockDim.x + threadIdx.x;

if (id < big_set_count)
{
int total = big_set_numbers[id];
for (int i = 0; i < tiny_set_count; i++)
{
total += tiny_shared[i];
}

big_set_numbers[id] *= total;
}
}