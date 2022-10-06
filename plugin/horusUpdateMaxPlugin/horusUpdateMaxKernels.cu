#include "customMultExamplePlugin.h"
#include <cuda_fp16.h>

namespace nvinfer1
{
namespace plugin
{

template <typename T>
__global__ void horusUpdateMaxKernel(const T* maxFrameIn, const T* toCompare, const int32_t* argmaxFrameIn, const int32_t convIdx, int numEls, 
                                     T* maxFrameOut, int32_t* argmaxFrameOut)
{
    int elIdx = blockIdx.x*blockDim.x + threadIdx.x;

    // probable that some threads left over on last block - these should do nothing
    if(elIdx < numEls)
    {
        if(toCompare[elIdx] > maxFrameIn[elIdx])
        {
            maxFrameOut[elIdx] = toCompare[elIdx];
            argmaxFrameOut[elIdx] = convIdx;
        }
        else
        {
            maxFrameOut[elIdx] = maxFrameIn[elIdx];
            argmaxFrameOut[elIdx] = argmaxFrameIn[elIdx];
        }
    }
}

template <typename T>
cudaError_t horusUpdateMaxKernelLauncher(const T* maxFrameIn, const T* toCompare, const int32_t* argmaxFrameIn, const int32_t convIdx, int iC, int iH, int iW, 
                                         T* maxFrameOut, int32_t* argmaxFrameOut, cudaStream_t stream)
{
    int numEls = iC*iH*iW;
    const int numThreadsPerBlock = 1024; // max threads per block for compute capability
    int numThreadBlocks = numEls/numThreadsPerBlock;
    if(numEls % numThreadsPerBlock != 0)
        numThreadBlocks++;

    horusUpdateMaxKernel<<<numThreadBlocks, numThreadsPerBlock, 0, stream>>>(maxFrameIn, toCompare, argmaxFrameIn, convIdx, numEls, maxFrameOut, argmaxFrameOut);

    return cudaPeekAtLastError();
}

template cudaError_t horusUpdateMaxKernelLauncher<float>(const float* maxFrame, const float* toCompare, const int32_t* argmaxFrame, const int32_t convIdx, int iC, int iH, int iW, 
                                                         float* maxFrameOut, int32_t* argmaxFrameOut, cudaStream_t stream);
template cudaError_t horusUpdateMaxKernelLauncher<__half>(const __half* maxFrame, const __half* toCompare, const int32_t* argmaxFrame, const int32_t convIdx, int iC, int iH, int iW, 
                                                          __half* maxFrameOut, int32_t* argmaxFrameOut, cudaStream_t stream);

} /* plugin */
} /* nvinfer1 */