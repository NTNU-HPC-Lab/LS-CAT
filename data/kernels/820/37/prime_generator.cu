#include "includes.h"
__global__ void prime_generator(int* d_input_list, uint64_cu* d_prime_list, uint64_cu* d_startPrimelist,uint64_cu* d_total_inputsize,uint64_cu* d_number_of_primes)
{

uint64_cu tid = (blockIdx.x*blockDim.x) + threadIdx.x;

if (tid < *d_number_of_primes) {
//            printf("Kaustubh\n");
uint64_cu primes=d_prime_list[tid];
//  printf("%llu\n",primes);
for(uint64_cu i=0;i<=d_total_inputsize[0];i++) // Added less than eual to here.
{
uint64_cu bucket= i/(WORD);
uint64_cu setbit= i%(WORD);
uint64_cu number=d_startPrimelist[0]+i;

//      printf("%llu -----> hash the value %llu to %llu bucket and change the %llu bit\n",number,i,bucket,setbit );
//      printf("**************  %llu --- %llu \n",number,primes);
if(number%primes==0)
{
//                                        printf("%llu is divisible by %llu \n", number,primes);
d_input_list[bucket]=d_input_list[bucket]| 1U<<setbit;
}
}
}
}