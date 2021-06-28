#include "includes.h"
__global__ void prime_generator(int* d_input_list, uint64_cu* d_prime_list, uint64_cu* d_startInputlist,uint64_cu* d_total_inputsize,uint64_cu* d_number_of_primes)
{

uint64_cu tid = (blockIdx.x*blockDim.x) + threadIdx.x;

//uint64_cu primes= d_prime_list[tid];
/*if(tid< d_number_of_primes[0])
printf("%d ---->  %llu\n",tid,primes);*/

//printf("THE NUMBER OF PRIMES ARE: %llu\n",*d_number_of_primes);
if (tid < *d_total_inputsize) {
//printf("Kaustubh\n");


uint64_cu actualNumber=*d_startInputlist+tid;
for(uint64_cu i=0;i<*d_number_of_primes;i++) { // Added less than eual to here.
uint64_cu bucket= tid/(WORD);
int setbit= tid%(WORD);

if(actualNumber%d_prime_list[i]==0) {
//printf("%llu is divisible by %llu \n", number,primes);
// THIS WAS WRONG  : d_input_list[bucket]=d_input_list[bucket]| 1U<<setbit;
atomicOr(&d_input_list[bucket],1U<<setbit); // setbit
break;
}
}
}
}