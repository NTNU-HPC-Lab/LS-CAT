#include "includes.h"

using namespace std;




__global__ void AddIntsCUDA(int* a, int* b) {
for (int i = 0; i < 12000000; i++)
{
a[0] += b[0];
}

}