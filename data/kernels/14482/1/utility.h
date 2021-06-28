#ifndef UTILITY_H
#define UTILITY_H

bool isNeg(char* bi);
__device__ __host__ int update(char* toUpdate, int value);
void init(int size, char* toFill);
int isFirstBiggerThanSecond(const char* first, const char* second, int size);
bool isFirstBiggerThanSecond_2(const char* first, const char* second, int size_f, int size_s);

#endif
