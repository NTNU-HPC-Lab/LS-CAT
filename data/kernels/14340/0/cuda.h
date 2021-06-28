
/* Functions in this file are meant to serve as drop-in CIVL replacements
 * for the Cuda function of the same name. Because of this, much of the
 * documentation of these functions is identical to the documentation
 * for its Cuda counterpart.
 */

#ifndef __CUDA
#define __CUDA

#include <civl-cuda.cvh>

/* Returns in *count the number of devices with compute capability 
 * greater or equal to 1.0 that are available for execution.
 */
cudaError_t cudaGetDeviceCount(int *count);

/* Returns in *device the current devie for the calling host thread
 */
cudaError_t cudaGetDevice(int * device);

/* Returns in *prop the properties of device dev
 */
cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp * prop, int dev);

/* Creates and event object
 */
cudaError_t cudaEventCreate(cudaEvent_t *event);

/* Records an event. If stream is non-zero, the event is recorded 
 * after all preceding operations in stream have been completed; 
 * otherwise, it is recorded after all preceding operations in the 
 * CUDA context have been completed. Since operation is asynchronous, 
 * cudaEventQuery() and/or cudaEventSynchronize() must be used to 
 * determine when the event has actually been recorded.
 */
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t s);

/* Query the status of all device work preceding the most recent call 
 * to cudaEventRecord() (in the appropriate compute streams, as 
 * specified by the arguments to cudaEventRecord()).
 * 
 * If this work has successfully been completed by the device, or if 
 * cudaEventRecord() has not been called on event, then cudaSuccess 
 * is returned. If this work has not yet been completed by the device 
 * then cudaErrorNotReady is returned.
 */
cudaError_t cudaEventQuery(cudaEvent_t event);


/* Wait until the completion of all device work preceding the most 
 * recent call to cudaEventRecord() (in the appropriate compute streams, 
 * as specified by the arguments to cudaEventRecord()).
 *
 * If cudaEventRecord() has not been called on event, cudaSuccess 
 * is returned immediately.
 */
cudaError_t cudaEventSynchronize(cudaEvent_t event);

/* since "timing" doesn't really make sense in the verification process
 * I'm not sure what this should do. maybe it shouldn't exist.
 */
cudaError_t cudaEventElapsedTime(float *t, cudaEvent_t from, cudaEvent_t to);

/* Destroys the event specified by event.
 */
cudaError_t cudaEventDestroy(cudaEvent_t event);

/* Creates a new asynchronous stream.
 */
cudaError_t cudaStreamCreate(cudaStream_t *pStream);


/* Blocks until stream has completed all operations.
 */
cudaError_t cudaStreamSynchronize(cudaStream_t stream);


/* Destroys and cleans up the asynchronous stream specified by stream.
 */
cudaError_t cudaStreamDestroy(cudaStream_t pStream);

/* Explicitly destroys and cleans up all resources associated with the 
 * current device in the current process. Any subsequent API call to 
 * this device will reinitialize the device.
 */
cudaError_t cudaDeviceReset( void );

/* locks until stream has completed all operations.
 */
cudaError_t cudaDeviceSynchronize( void );

/* Copies count bytes from the memory area pointed to by src to the 
 * memory area pointed to by dst, where kind is one of 
 * cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, 
 * or cudaMemcpyDeviceToDevice, and specifies the direction of the 
 * copy. The memory areas may not overlap.
 */
cudaError_t cudaMemcpy ( void *dst, const void *src, size_t count, enum cudaMemcpyKind kind );

/* Not implemented. Prototype provided for compiling purposes.
 */
cudaError_t cudaMalloc( void *ptr, size_t size);

/* Fills the first count bytes of the memory area pointed to by devPtr 
 * with the constant byte value value
 */
cudaError_t cudaMemset(void * devPtr, int value, size_t count);

/* Frees the memory space pointed to by devPtr. Similar semantics to free/$free.
 */
cudaError_t cudaFree(void *devPtr);

/* Sets device as the current device for the calling host thread. Currently,
 * only a single device is supported, so this call always succeeds with a noop.
 */
cudaError_t cudaSetDevice(int device_id);

/* Returns the message string from an error code
 */
const char* cudaGetErrorString(cudaError_t error);

/* Returns the last error that has been produces by any of the runtime calls 
 * in the same host thread and resets it to cudaSuccess
 */
cudaError_t cudaGetLastError(void);

/* DEPRECATED. DO NOT USE
 */
cudaError_t cudaThreadExit(void);

/* Not implemented. Prototype provided for compatibilty purposes
 */
void __syncthreads( void );

uint3 threadIdx;
uint3 blockIdx;
dim3 gridDim;
dim3 blockDim;

#endif
