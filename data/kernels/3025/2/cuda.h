#ifndef CUDA_HELPER
#define CUDA_HELPER
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

typedef unsigned int uint;
typedef unsigned long ulong;
typedef unsigned char uchar;

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
#define MAX_ARRAYS 100


struct Dimensions {
    int height;
    int width;
    int sizeofElement;
};

struct TextureWrapper {
    size_t pitch;
    void** devicePointer;
};

struct CudaContext {
    uint* sizes;
    uchar* isOutput;
    uchar* createdByContext;
    struct Dimensions* dimensions;
    void** devicePointers;
    void** hostPointers;
    void*** twoDimensionalHostPointers;
    
    int cudaPointerCount = 0;
    int deviceCount;

    void init() {
        //HANDLE_ERROR(cuInit(0));
        HANDLE_ERROR(cudaGetDeviceCount(&deviceCount));
        devicePointers = (void**) malloc(sizeof(void*)*MAX_ARRAYS);
        hostPointers = (void**) malloc(sizeof(void*)*MAX_ARRAYS);
        twoDimensionalHostPointers = (void***) malloc(sizeof(void**)*MAX_ARRAYS);
        sizes = (uint*) malloc(sizeof(uint)*MAX_ARRAYS);
        isOutput = (uchar*) malloc(sizeof(uchar)*MAX_ARRAYS);
        createdByContext = (uchar*) malloc(sizeof(uchar)*MAX_ARRAYS);
        dimensions = (struct Dimensions*) malloc(sizeof(struct Dimensions)*MAX_ARRAYS);
        memset(isOutput,0,sizeof(uchar)*MAX_ARRAYS);
        memset(createdByContext,0,sizeof(uchar)*MAX_ARRAYS);
        memset(sizes,0,sizeof(uint)*MAX_ARRAYS);

        for (int i=0;i<MAX_ARRAYS;i++) {
            struct Dimensions newDimensions;
            newDimensions.height = 0;
            newDimensions.width = 0;
            newDimensions.sizeofElement = 0;
            dimensions[i] = newDimensions;
        }
    }

    int getBlocks(uint N) {
        return ceil((N+1024)/1024);
    }

    dim3 getBlocks(int rows,int columns) {
        dim3 blocks(ceil((rows+32)/32),ceil((columns+32)/32));
        return blocks;
    }

    dim3 getBlocks(int rows,int columns,int blockSize) {
        dim3 blocks(ceil((rows+blockSize)/blockSize),ceil((columns+blockSize)/blockSize));
        return blocks;
    }

    int getThreads(uint N) {
        return 1024;
    }

    dim3 getThreads(int rows,int columns) {
        dim3 threads(32,32);
        return threads;
    }

    void cudaInConstant(void* hostData, void** deviceData,uint sizeInBytes) {
        HANDLE_ERROR(cudaMemcpyToSymbol(*deviceData,hostData,sizeInBytes));
    }

    struct TextureWrapper cudaInTexture(texture<float,2,cudaReadModeElementType> *tex_w,void** hostData,int width,int height,int sizeOfElement) {
        void* tex_arr;
        size_t pitch;
        uint widthSize = width*sizeOfElement;
        HANDLE_ERROR( cudaMallocPitch((void**)&tex_arr, &pitch, widthSize, height) );
        HANDLE_ERROR( cudaMemcpy2D(tex_arr,             // device destination                                   
                            pitch,           // device pitch (calculated above)                      
                            hostData,               // src on host                                          
                            widthSize, // pitch on src (no padding so just width of row)       
                            widthSize, // width of data in bytes                               
                            height,            // height of data                                       
                            cudaMemcpyHostToDevice) );
               
        tex_w->normalized = false;  // don't use normalized values                                           
        tex_w->filterMode = cudaFilterModeLinear;
        tex_w->addressMode[0] = cudaAddressModeClamp; // don't wrap around indices                           
        tex_w->addressMode[1] = cudaAddressModeClamp;
        //  HANDLE_ERROR( cudaBindTexture2D(NULL,  tex_w, *wrapper.devicePointer,  tex_w.channelDesc, 2, 2, wrapper.pitch) );  use this in calling 
        // HANDLE_ERROR(cudaUnbindTexture(tex_w));
        struct TextureWrapper wrapper;
        wrapper.pitch = pitch;
        wrapper.devicePointer = &tex_arr;

        devicePointers[cudaPointerCount] = tex_arr;
        hostPointers[cudaPointerCount] = (void*) malloc(1);
        createdByContext[cudaPointerCount] = 1;
        cudaPointerCount++;
        return wrapper;
    }

    void* cudaIn(void* hostData,uint sizeInBytes) {
        void* deviceData;
        HANDLE_ERROR(cudaMalloc((void**)&deviceData,sizeInBytes));
        HANDLE_ERROR(cudaMemcpy(deviceData,hostData,sizeInBytes,cudaMemcpyHostToDevice));
        devicePointers[cudaPointerCount] = deviceData;
        hostPointers[cudaPointerCount] = hostData;
        //printf("device pointer:%p host pointer:%p size:%u\n",deviceData,hostData,sizeInBytes);
        sizes[cudaPointerCount] = sizeInBytes;
        cudaPointerCount++;
        return deviceData;
    }

    void* cudaInOut(void* hostData,uint sizeInBytes) {
        isOutput[cudaPointerCount] = 1;
        return cudaIn(hostData,sizeInBytes);
    }

    void* cudaIn(void** hostData,int elementSize,int width,int height) {
        uint size = width*height*elementSize;
        uint widthSize = width*elementSize;
        
        char** hostDataAsChar = (char**) hostData;
        void* hostDataFlattened = (void*) malloc(size);
        char* hostDataFlattenedAsChar = (char*) hostDataFlattened;

        for (int y=0;y<height;y++) {
            char* rowData = hostDataAsChar[y];
            memcpy(&hostDataFlattenedAsChar[y*widthSize],rowData,widthSize);
        }

        struct Dimensions currentDimension = dimensions[cudaPointerCount];
        currentDimension.height = height;
        currentDimension.width = width;
        currentDimension.sizeofElement = elementSize;
        dimensions[cudaPointerCount] = currentDimension;

        twoDimensionalHostPointers[cudaPointerCount] = hostData;

        createdByContext[cudaPointerCount] = 1;

        return cudaIn(hostDataFlattened,size);
    }

    void* cudaInOut(void** hostData,int elementSize,int width,int height) {
        isOutput[cudaPointerCount] = 1;
        return cudaIn(hostData,elementSize,width,height);
    }

    void synchronize() {
        synchronize(NULL);
    }

    void synchronize(void** hostData,int width,int height) {
       for (int i=0;i<cudaPointerCount;i++) {
           if (twoDimensionalHostPointers[i]==hostData) {
               synchronize(hostPointers[i]);
           }
       }
    }

    void synchronize(void* ptr) {
        cudaDeviceSynchronize();
        for (int i=0;i<cudaPointerCount;i++) {
            if (ptr==NULL || ptr==hostPointers[i]) {
                if (isOutput[i]) {
                    //printf("%d: device pointer:%p host pointer:%p size:%u\n",i,devicePointers[i],hostPointers[i],sizes[i]);
                    HANDLE_ERROR(cudaMemcpy(hostPointers[i],devicePointers[i],sizes[i],cudaMemcpyDeviceToHost));
                    struct Dimensions currentDimension = dimensions[i];
                    if (currentDimension.width>0 && currentDimension.height>0 && currentDimension.sizeofElement>0) {
                        char** mtx = (char**) twoDimensionalHostPointers[i];
                        char* hostPointerAsChar = (char*) hostPointers[i];
                        uint rowSize = currentDimension.width*currentDimension.sizeofElement;

                        for (int i=0;i<currentDimension.height;i++) {
                            char* row = mtx[i];
                            memcpy(row,&hostPointerAsChar[i*rowSize],rowSize);  
                        }
                    }
                }
            } 
        }

        for (int i=0;i<cudaPointerCount;i++) {
            if (ptr==NULL || ptr==hostPointers[i]) {
                if (!isOutput[i] && !createdByContext[i]) {
                    HANDLE_ERROR(cudaMemcpy(devicePointers[i],hostPointers[i],sizes[i],cudaMemcpyHostToDevice)); 
                }
            }
        }
    }

    void displayProperties() {
        for (int i=0;i<deviceCount;i++) {
            cudaDeviceProp prop;
            HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));
            printf("Device Number: %d\n", i);
            printf("  Device name: %s\n", prop.name);
            printf("  Memory Clock Rate (KHz): %d\n",
                prop.memoryClockRate);
            printf("  Memory Bus Width (bits): %d\n",
                prop.memoryBusWidth);
            printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
                2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
        }
    }

    void dispose() {
        for (int i=0;i<cudaPointerCount;i++) {
            cudaFree(devicePointers[i]);
            if (createdByContext[i]) {
                free(hostPointers[i]);//we added this to memory
            }
        }

        free(createdByContext);
        free(sizes);
        free(isOutput);
        free(dimensions);
        free(hostPointers);
        free(twoDimensionalHostPointers);
        free(devicePointers);
    }
};

#endif
