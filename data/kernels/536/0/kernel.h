#pragma once
cudaError_t CopyDataToDevice(unsigned frameCount, unsigned char** allImageDataOnHost, unsigned char** allImageDataOnDevice, unsigned width, unsigned height);