#pragma once

#include <memory>
#include "error_handling.hpp"

/*
 * A Matrix class which can have the same matrix on both the CPU and GPU.
 * The functions like cudaMemcpy(in either direction), cudaMalloc, cudaFree
 * have been abstracted away in this class for convenience.
 */
class Matrix
{
 public:
  Matrix()
  {
    /* Nothing to do here */
  }

  Matrix(int nRows, int nCols) :
    nRows(nRows),
    nCols(nCols),
    memAllocated(false)
  {
    AllocateMemory();
  }

  void AllocateMemory()
  {
    if (!memAllocated)
    {
      // Allocate memory on the host.
      hostMat = std::shared_ptr<float>(new float[nRows * nCols],
          [&]/* lambda function. */(float* ptr){delete[] ptr;});

      // Allocate memory on the device.
      float* deviceMem = nullptr;
      CheckErrors(cudaMalloc(&deviceMem, nRows * nCols * sizeof(float)),
          "Matrix::AllocateMemory() cudaMalloc : deviceMem");

      deviceMat = std::shared_ptr<float>(deviceMem,
          [&]/* lambda function. */(float* ptr){cudaFree(ptr);});

      memAllocated = true;
    }
  }

  void AllocateMemory(int nRows, int nCols)
  {
    if (!memAllocated)
    {
      this->nRows = nRows;
      this->nCols = nCols;

      AllocateMemory();
    }
  }

  void CopyDeviceToHost()
  {
    if (memAllocated)
    {
      CheckErrors(cudaMemcpy(hostMat.get(), deviceMat.get(),
          nRows * nCols * sizeof(float), cudaMemcpyDeviceToHost),
          "Matrix::CopyDeviceToHost() cudaMemcpy : deviceMat");
    }
    else
    {
      std::cerr << "ERROR: Cannot copy device matrix data to host if memory " <<
          "is not allocated." << std::endl;
      exit(-1);
    }
  }

  void CopyHostToDevice()
  {
    if (memAllocated)
    {
      CheckErrors(cudaMemcpy(deviceMat.get(), hostMat.get(),
          nRows * nCols * sizeof(float), cudaMemcpyHostToDevice),
          "Matrix::CopyHostToDevice cudaMemcpy : deviceMat");
    }
    else
    {
      std::cerr << "ERROR: Cannot copy host matrix data to device if memory " <<
          "is not allocated." << std::endl;
      exit(-1);
    }
  }

  float& operator[](const int index)
  {
    // Uncomment when debugging.
    // if (index >= nRows * nCols)
    // {
    //   std::cerr << "ERROR: Matrix::operator() index out of bounds" << std::endl;
    //   exit(-1);
    // }
    return hostMat.get()[index];
  }

  // Overload (row, col) for 2D indexing.
  float& operator()(const int row, const int col)
  {
    // Uncomment when debugging.
    // if (row >= nRows || col >= nCols)
    // {
    //   std::cerr << "ERROR: Matrix::operator() index out of bounds" << std::endl;
    //   exit(-1);
    // }

    return hostMat.get()[row * nCols + col];
  }

  std::shared_ptr<float> deviceMat;
  std::shared_ptr<float> hostMat;

  int nRows, nCols;

 private:
  bool memAllocated;
};
