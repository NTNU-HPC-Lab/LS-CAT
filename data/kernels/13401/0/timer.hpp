#ifndef TIMER_HPP_
#define TIMER_HPP_

#include "common.hpp"

class Timer {
 public:
  Timer() {
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));
  }

  virtual ~Timer() {
    CUDA_CHECK(cudaEventDestroy(start_));
    CUDA_CHECK(cudaEventDestroy(stop_));
  }

  virtual void Start() {
    CUDA_CHECK(cudaEventRecord(start_, 0));
  }

  virtual void Stop() {
    CUDA_CHECK(cudaEventRecord(stop_, 0));
  }

  virtual float Milliseconds() {
    CUDA_CHECK(cudaEventSynchronize(stop_));
    float msec;
    CUDA_CHECK(cudaEventElapsedTime(&msec, start_, stop_));
    return msec;
  }

 private:
  cudaEvent_t start_;
  cudaEvent_t stop_;
};

#endif