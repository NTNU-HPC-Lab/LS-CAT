#include <iostream>

void CheckErrors(cudaError_t error, char const *message)
{
  if (error != cudaSuccess)
  {
    // fprintf(stderr, "ERROR: %s : %s\n", message, cudaGetErrorString(error));
    std::cerr << "ERROR: " << message << " : " << cudaGetErrorString(error)
       << std::endl;
    exit(-1);
  }
}
