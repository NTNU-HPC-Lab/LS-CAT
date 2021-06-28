#pragma once
#include <cstddef>

#ifdef __cplusplus
extern "C"
{
#endif

#ifndef __GNUC__

#ifdef SEBA_EXPORTS
#define DLL __declspec(dllexport) __cdecl
#else
#define DLL
#endif

#else

#define DLL

#endif

	typedef struct sebaRawUnpackerHandleStruct *sebaRawUnpackerHandle_t;
	typedef struct sebaDeviceSurfaceBuffer_t *sebaDeviceSurfaceBufferHandle_t;
	typedef struct sebaDebayerHandleStruct *sebaDebayerHandle_t;
	typedef struct sebaSurfaceConverterHandleStruct *sebaSurfaceConverterHandle_t;
	typedef struct sebaExportToHostStruct *sebaExportToHostHandle_t;
	typedef struct sebaImportFromHostStruct *sebaImportFromHostHandle_t;
	typedef struct sebaEncoderHandleStruct *sebaJpegEncoderHandle_t;

	typedef enum
	{
		SEBA_OK, // There is no error during function execution

		SEBA_TRIAL_PERIOD_EXPIRED,

		SEBA_INVALID_DEVICE,			  // Device with selected index does not exist or device is non NVIDIA device or device is non CUDA-compatible device
		SEBA_INCOMPATIBLE_DEVICE,		  // Device is CUDA-compatible, but its compute compatibility is below 2.0, thus device is considered to be incompatible with SDK
		SEBA_INSUFFICIENT_DEVICE_MEMORY, // Available device memory is not enough to allocate new buffer
		SEBA_INSUFFICIENT_HOST_MEMORY,   // Available host memory is not enough to allocate new buffer
		SEBA_INVALID_HANDLE,			  // Component handle is invalid or has inappropriate type
		SEBA_INVALID_VALUE,			  // Some parameter of the function called is invalid or combination of input parameters are unacceptable
		SEBA_UNAPPLICABLE_OPERATION,	 // This operation can not be applied to the current type of data
		SEBA_INVALID_SIZE,				  // Image dimension is invalid
		SEBA_UNALIGNED_DATA,			  // Buffer base pointers or pitch are not properly aligned
		SEBA_INVALID_TABLE,			  // Invalid quantization / Huffman table
		SEBA_BITSTREAM_CORRUPT,		  // JPEG bitstream is corrupted and can not be decoded
		SEBA_EXECUTION_FAILURE,		  // Device kernel execution failure
		SEBA_INTERNAL_ERROR,			  // Internal error, non-kernel software execution failure
		SEBA_UNSUPPORTED_SURFACE,

		SEBA_IO_ERROR,			  // Failed to read/write file
		SEBA_INVALID_FORMAT,	 // Invalid file format
		SEBA_UNSUPPORTED_FORMAT, // File format is not supported by the current version of SDK
		SEBA_END_OF_STREAM,

		SEBA_MJPEG_THREAD_ERROR,
		SEBA_TIMEOUT,
		SEBA_MJPEG_OPEN_FILE_ERROR,

		SEBA_UNKNOWN_ERROR // Unrecognized error
	} sebaStatus_t;

	typedef enum
	{
		SEBA_I8,
		SEBA_I10,
		SEBA_I12,
		SEBA_I14,
		SEBA_I16,

		SEBA_RGB8,
		SEBA_BGR8,
		SEBA_RGB12,
		SEBA_RGB16,

		SEBA_BGRX8,

		SEBA_CrCbY8,
		SEBA_YCbCr8
	} sebaSurfaceFormat_t;

	typedef enum
	{
		SEBA_RAW_XIMEA12,
		SEBA_RAW_PTG12,
	} sebaRawFormat_t;

	typedef enum
	{
		SEBA_DFPD,
		SEBA_HQLI,
		SEBA_MG,
		SEBA_DFPD_UNREFINED,
	} sebaDebayerType_t;

	typedef enum
	{
		SEBA_BIT_DEPTH,
		SEBA_SELECT_CHANNEL
	} sebaSurfaceConverter_t;

	typedef enum
	{
		SEBA_BAYER_NONE,
		SEBA_BAYER_RGGB,
		SEBA_BAYER_BGGR,
		SEBA_BAYER_GBRG,
		SEBA_BAYER_GRBG,
	} sebaBayerPattern_t;

	typedef enum
	{
		SEBA_CHANNEL_R,
		SEBA_CHANNEL_G,
		SEBA_CHANNEL_B
	} sebaChannelType_t;

	typedef enum
	{
		SEBA_CONVERT_NONE,
		SEBA_CONVERT_BGR
	} sebaConvertType_t;

	typedef enum
	{
		SEBA_SEQUENTIAL_DCT,
		SEBA_LOSSLESS
	} sebaJpegMode_t;

	typedef enum
	{
		SEBA_JPEG_Y,
		SEBA_JPEG_444,
		SEBA_JPEG_422,
		SEBA_JPEG_420
	} sebaJpegFormat_t;

	typedef struct
	{
		sebaChannelType_t channel;
	} sebaSelectChannel_t;

	typedef struct
	{
		sebaConvertType_t convert;
	} sebaExportParameters_t;

	typedef struct
	{
		unsigned surfaceFmt;
		unsigned width;
		unsigned height;
		unsigned pitch;
		unsigned maxWidth;
		unsigned maxHeight;
		unsigned maxPitch;
	} sebaDeviceSurfaceBufferInfo_t;

	typedef struct
	{
		unsigned short exifCode;
		char *exifData;
		int exifLength;
	} sebaJpegExifSection_t;

	typedef struct
	{
		unsigned short data[64];
	} sebaQuantTable_t;

	typedef struct
	{
		sebaQuantTable_t table[4];
	} sebaJpegQuantState_t;

	typedef struct
	{
		unsigned char bucket[16];
		unsigned char alphabet[256];
	} sebaHuffmanTable_t;

	typedef struct
	{
		sebaHuffmanTable_t table[2][2];
	} sebaJpegHuffmanState_t;

	typedef struct
	{
		unsigned quantTableMask;
		unsigned huffmanTableMask[2];
		unsigned scanChannelMask;
		unsigned scanGroupMask;
	} sebaJpegScanStruct_t;

	typedef struct
	{
		sebaJpegMode_t jpegMode;
		sebaJpegFormat_t jpegFmt;

		int predictorClass;

		unsigned char *h_Bytestream;
		unsigned bytestreamSize;
		unsigned headerSize;

		unsigned height;
		unsigned width;
		unsigned bitsPerChannel;

		sebaJpegExifSection_t *exifSections;
		unsigned exifSectionsCount;

		sebaJpegQuantState_t quantState;
		sebaJpegHuffmanState_t huffmanState;
		sebaJpegScanStruct_t scanMap;
		unsigned restartInterval;
	} sebaJfifInfo_t;

	extern sebaStatus_t DLL sebaGetDeviceSurfaceBufferInfo(
		sebaDeviceSurfaceBufferHandle_t buffer,
		sebaDeviceSurfaceBufferInfo_t *devBuffer);

	extern sebaStatus_t DLL sebaRawUnpackerCreate(
		sebaRawUnpackerHandle_t *handle,

		sebaRawFormat_t rawFmt,
		sebaSurfaceFormat_t surfaceFmt,

		unsigned maxWidth,
		unsigned maxHeight,

		sebaDeviceSurfaceBufferHandle_t *dstBuffer);

	extern sebaStatus_t DLL sebaDebayerCreate(
		sebaDebayerHandle_t *handle,

		sebaDebayerType_t debayerType,

		unsigned maxWidth,
		unsigned maxHeight,

		sebaDeviceSurfaceBufferHandle_t srcBuffer,
		sebaDeviceSurfaceBufferHandle_t *dstBuffer);

	extern sebaStatus_t DLL sebaJpegEncoderCreate(
		sebaJpegEncoderHandle_t *handle,

		unsigned maxWidth,
		unsigned maxHeight,

		sebaDeviceSurfaceBufferHandle_t srcBuffer);

	extern sebaStatus_t DLL sebaSurfaceConverterCreate(
		sebaSurfaceConverterHandle_t *handle,

		sebaSurfaceConverter_t surfaceConverterType,
		void *staticSurfaceConverterParameters,

		unsigned maxWidth,
		unsigned maxHeight,

		sebaDeviceSurfaceBufferHandle_t srcBuffer,
		sebaDeviceSurfaceBufferHandle_t *dstBuffer);

	extern sebaStatus_t DLL sebaExportToHostCreate(
		sebaExportToHostHandle_t *handle,
		sebaSurfaceFormat_t *surfaceFmt,

		sebaDeviceSurfaceBufferHandle_t srcBuffer);

	extern sebaStatus_t DLL sebaRawUnpackerDecode(
		sebaRawUnpackerHandle_t handle,

		void *h_src,
		unsigned width,
		unsigned height);

	extern sebaStatus_t DLL sebaDebayerTransform(
		sebaDebayerHandle_t handle,

		sebaBayerPattern_t bayerFmt,
		unsigned width,
		unsigned height);

	extern sebaStatus_t DLL sebaSurfaceConverterTransform(
		sebaSurfaceConverterHandle_t handle,
		void *surfaceConverterParameters,

		unsigned width,
		unsigned height);

	extern sebaStatus_t DLL sebaExportToHostCopy(
		sebaExportToHostHandle_t handle,
		void *h_dst,

		unsigned width,
		unsigned pitch,
		unsigned height,

		sebaExportParameters_t *parameters);

	extern sebaStatus_t DLL sebaJpegEncode(
		sebaJpegEncoderHandle_t handle,

		unsigned quality,

		sebaJfifInfo_t *jfifInfo);

	extern sebaStatus_t DLL sebaRawUnpackerDestroy(sebaRawUnpackerHandle_t handle);
	extern sebaStatus_t DLL sebaExportToHostDestroy(sebaExportToHostHandle_t handle);
	extern sebaStatus_t DLL sebaDebayerDestroy(sebaDebayerHandle_t handle);
	extern sebaStatus_t DLL sebaJpegEncoderDestroy(sebaJpegEncoderHandle_t handle);
	extern sebaStatus_t DLL sebaSurfaceConverterDestroy(sebaSurfaceConverterHandle_t handle);

	extern sebaStatus_t DLL sebaImportFromHostCreate(
		sebaImportFromHostHandle_t *handle,
		sebaSurfaceFormat_t surfaceFmt,

		unsigned maxWidth,
		unsigned maxHeight,

		sebaDeviceSurfaceBufferHandle_t *dstBuffer);

	extern sebaStatus_t DLL sebaImportFromHostCopy(
		sebaImportFromHostHandle_t handle,

		void *h_src,
		unsigned width,
		unsigned pitch,
		unsigned height);

	extern sebaStatus_t DLL sebaImportFromHostDestroy(sebaImportFromHostHandle_t handle);

	extern sebaStatus_t DLL sebaMalloc(void **buffer, size_t size);
	extern sebaStatus_t DLL sebaFree(void *buffer);

#ifdef __cplusplus
}
#endif
