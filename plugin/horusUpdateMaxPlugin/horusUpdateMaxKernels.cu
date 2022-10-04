#include "customMultExamplePlugin.h"
#include <cuda_fp16.h>

namespace nvinfer1
{
namespace plugin
{

template <typename T>
__global__ void horusUpdateMaxKernel(T* maxFrame, const T* toCompare, int32_t* argmaxFrame, const int32_t convIdx, int numEls)
{
    int elIdx = blockIdx.x*blockDim.x + threadIdx.x;

    // probable that some threads left over on last block - these should do nothing
    if(elIdx < numEls)
    {
        if(toCompare[elIdx] > maxFrame[elIdx])
        {
            maxFrame[elIdx] = toCompare[elIdx];
            argmaxFrame[elIdx] = convIdx;
        }
    }
}

template <typename T>
cudaError_t horusUpdateMaxKernelLauncher(T* maxFrame, const T* toCompare, int32_t* argmaxFrame, const int32_t convIdx, int iC, int iH, int iW, cudaStream_t stream)
{
    int numEls = iC*iH*iW;
    const int numThreadsPerBlock = 1024; // max threads per block for compute capability
    int numThreadBlocks = numEls/numThreadsPerBlock;
    if(numEls % numThreadsPerBlock != 0)
        numThreadBlocks++;

    horusUpdateMaxKernel<<<numThreadBlocks, numThreadsPerBlock, 0, stream>>>(maxFrame, toCompare, argmaxFrame, convIdx, numEls);

    return cudaPeekAtLastError();
}

template cudaError_t horusUpdateMaxKernelLauncher<float>(float* maxFrame, const float* toCompare, int32_t* argmaxFrame, const int32_t convIdx, int iC, int iH, int iW, cudaStream_t stream);
template cudaError_t horusUpdateMaxKernelLauncher<__half>(__half* maxFrame, const __half* toCompare, int32_t* argmaxFrame, const int32_t convIdx, int iC, int iH, int iW, cudaStream_t stream);

} /* plugin */
} /* nvinfer1 */