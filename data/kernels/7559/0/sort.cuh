#ifndef SORT
#define SORT

void mergeSort(int *h_array, int arraySize, double *networkTime);
__global__ void gpu_sort(int *d_array, int arraySize, int chunkSize);
__global__ void gpu_merge(int *d_array, int *d_temp_array, int arraySize, int chunkSize);
void cpuMerge(int *data, int size, int chunkSize);
__host__ __device__ void mergeArrays(int *data, int *buffer, int a, int m, int b);
__device__ void insertionSort(int *array, int a, int b);
__host__ __device__ void printArray(int *d_array, int size);
int *getRandomArray(int size, int startRange, int endRange);
int randInt(int a, int b);
int comparator(const void *p, const void *q);
int compareArrays(int *array1, int *array2, int size);

#endif
