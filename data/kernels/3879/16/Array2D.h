#pragma once
#include <vector>
#include <stdio.h>

template <typename T>
class Array2D
{
  private:
    std::vector<T> data;
    size_t rowCount;
    size_t colCount;

  public:
    Array2D(size_t rowCount, size_t colCount)
    {
        this->rowCount = rowCount;
        this->colCount = colCount;

        data.resize(rowCount * colCount);
    }
    ~Array2D()
    {
        data.clear();
    }

    T *data_ptr()
    {
        return this->data.data();
    }

    T &at(size_t row, size_t col)
    {
        return data[(row * colCount) + col];
    }
    const T &at(size_t row, size_t col) const
    {
        return data[(row * colCount) + col];
    }

    void print() const
    {
        for (size_t row = 0; row < rowCount; row++)
        {
            for (size_t col = 0; col < colCount; col++)
            {
                printf("%5i ", at(row, col));
            }
            printf("\n");
        }
    }
};