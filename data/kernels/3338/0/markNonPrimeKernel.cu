#include "includes.h"



#define number_type unsigned long long

const int block_size = 1024; // 2**10 threads
const int thread_size = 32768 * 2 * 2; // 2**15 max elements per thread always keep even number
const number_type max_chunk_size = pow(2, 31) + pow(2, 30); // 2**31 items cause reduce ram use else failed allocations, always keep even number

cudaError_t find_primes_cuda(number_type n, number_type r);

void set_one(char* dev_arr, unsigned int size);
template <typename T>
void reset(T* dev_arr, size_t count);

template <typename T>
T* device(size_t count);
template <typename T>
T* host(size_t count);
void confirmCudaNoError();
void cudaWait();
template <typename T>
T* to_host(const T* dev_ptr, size_t count, T* host_ptr = nullptr);
template <typename T>
T* to_device(const T* host_ptr, size_t count, T* dev_ptr = nullptr);



//__global__ void markNonPrimeKernel(char* dev_chunk, number_type* min_primes, number_type currentValue, number_type currentValueSqr,
//	const number_type startValue, const number_type endValue, const int thread_size)
//{
//	const auto myThreadId = blockIdx.x * block_size + threadIdx.x;
//	const auto myStartValue = startValue + myThreadId * thread_size;
//	auto myEndValue = myStartValue + thread_size;
__global__ void markNonPrimeKernel(char* dev_chunk, number_type currentValue, number_type currentValueSqr, const number_type startValue, const number_type endValue, const int thread_size)
{
const auto myThreadId = blockIdx.x * block_size + threadIdx.x;
const auto myStartValue = startValue + myThreadId * thread_size;
auto myEndValue = myStartValue + thread_size;
if (myEndValue > endValue)
{
myEndValue = endValue;
}

number_type offset = 1;
// if current min first is set then we can offset by currentValue but if
// the number i is odd (which we can make sure of) then we can increment by
// currentValue * 2 as then we skip all even numbers in between which we dont need anyway
// as they will be already marked in case of 2
const int offsetMultiplier = (currentValue == 2) ? 1 : 2; //

auto updated_start = myStartValue;
if (updated_start != 0) // in case of zero first statement will underflow and will lead to max value
{
updated_start = myStartValue - myStartValue % currentValue;
if (updated_start % 2 == 0) // if even make it odd as only odd numbers can be marked off
//(even are done in case of 2, in which case subtracting 2 will still make it even)
{
updated_start -= currentValue;
}
}

if (updated_start < currentValueSqr)
updated_start = currentValueSqr;
offset = currentValue * offsetMultiplier;

for (auto i = updated_start; i < myEndValue; i += offset)
{
dev_chunk[i - startValue] = 0; // cancel that number, min is already marked, offset is current number
}
}