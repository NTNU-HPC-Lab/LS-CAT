#include "includes.h"
__global__ void prime_generator(int* d_input_list, uint64_cu* d_prime_list, uint64_cu* d_startPrimelist,uint64_cu* d_total_inputsize,uint64_cu* d_number_of_primes)
{

uint64_cu tid = (blockIdx.x*blockDim.x) + threadIdx.x;

//uint64_cu primes= d_prime_list[tid];
/*if(tid< d_number_of_primes[0])
printf("%d ---->  %llu\n",tid,primes);*/

//printf("THE NUMBER OF PRIMES ARE: %llu\n",*d_number_of_primes);
if (tid < *d_number_of_primes) {
//printf("Kaustubh\n");
uint64_cu primes=d_prime_list[tid];
for(uint64_cu i=0;i<d_total_inputsize[0];i++) { // Added less than eual to here.
uint64_cu bucket= i/(WORD);
int setbit= i%(WORD);
uint64_cu number=d_startPrimelist[0]+i;
//printf("THE NUMBER %llu IS BEING DIVIDED BY %llu\n",number,primes);
if(number%primes==0) {
//printf("%llu is divisible by %llu \n", number,primes);
// THIS WAS WRONG  : d_input_list[bucket]=d_input_list[bucket]| 1U<<setbit;
if(0 == (d_input_list[bucket] & 1U<<setbit)){ // testbit
atomicOr(&d_input_list[bucket],1U<<setbit); // setbit
}
}
}
}
}