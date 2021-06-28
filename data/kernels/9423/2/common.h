#ifndef COMMON_CUH
#define COMMON_CUH

#include <vector>
#include <iostream>

#define MAX_BLOCK_SIZE 1024

void Task1();

void Task2();

void Task3();

bool GoodMiltiplication(std::vector<unsigned> const& a, std::vector<unsigned> const& b, std::vector<unsigned> const& result);

template <typename T>
void WriteVector(std::vector<T> const& values, std::ostream& out)
{
    for (auto const& item: values) {
        out << item << std::endl;
    }
}

#endif // COMMON_CUH
