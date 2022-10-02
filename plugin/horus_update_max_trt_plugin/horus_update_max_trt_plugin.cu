#include "horus_update_max_trt_plugin.h"
#include <cuda_fp16.h>

// might need to handle sending data to gpu

const int num_threads_per_block = 1024; // max threads per block for compute capability

/**
 * Computes optimal number of thread blocks (1 thread per array element)
 *
 * @param   total_els   total number of elements to process; i.e., total number of threads to launch
 * @return  num_thread_blocks   the number of threads blocks to use in order to accomodate number of needed threads
 */
int get_num_thread_blocks_trt(int total_els)
{
    int num_thread_blocks = total_els/num_threads_per_block;
    if(total_els % num_threads_per_block != 0)
    {
        num_thread_blocks+=1;
    }

    return num_thread_blocks;
}

/**
 * Kernel implementation of custom PyTorch op horus_update_max
 *
 * @param[out]  max_frame       current array of maximum pixel intensities, to be modified in place
 * @param       to_compare      array of potential new maximum values to compare with max_frame
 * @param[out]  arg_max_frame   array of the same size as max_frame that keeps track of which conv a pixel's max came from, to be modified in place
 * @param       conv_idx        index to place in argmax_frame wherever a new maximum value is found
 * @param       num_els         number of total elements to process (number of elements in max_frame)
 */
template<typename T_DATA>
__global__ 
void horus_update_max_kernel_trt(T_DATA* max_frame, T_DATA* to_compare, int32_t* argmax_frame, int conv_idx, int num_els)
{
    int el_idx = blockIdx.x*blockDim.x + threadIdx.x;

    // probable that some threads left over on last block - these should do nothing
    if(el_idx < num_els)
    {
        if( conv_idx == 0 || *(to_compare + el_idx) > *(max_frame + el_idx) )
        {
            *(max_frame + el_idx) = *(to_compare + el_idx);
            *(argmax_frame + el_idx) = conv_idx;
        }
    }
    __syncthreads();
}

template <typename T>
int inference_horus_update_max(int batch_size, , int iC, int iH, int iW, T* max_frame, T* to_compare, int32_t* argmax_frame, int conv_idx, int num_els_per_image, cudaStream_t stream)
{
    int num_els = iC*iH*iW;
    int num_thread_blocks = get_num_thread_blocks_trt(num_els);
    for (int i = 0; i < batchSize; ++i)
    {
        // NOTE: kernelCopy kernel can be replaced with cudaMemcpy function
        // kernelCopy<<<nBlocksCopy, nThreads, 0, stream>>>(lenCopy, inputs, outputs);
        // outputs += lenCopy;
        
        horus_update_max_kernel_trt<<<num_thread_blocks, num_threads_per_block, 0, stream>>>(max_frame, to_compare, argmax_frame, conv_idx, num_els_per_image);
        max_frame += num_els_per_image;
        to_compare += num_els_per_image;
        argmax_frame += num_els_per_image;
    }

    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int horus_update_max_trt_plugin::enqueue(
    int batchSize, void* const* max_frame, const void* const* to_compare, void* const* argmax_frame, void* workspace, cudaStream_t stream) noexcept
{
    switch (iType)
    {
    case DataType::kFLOAT:
        return inference_horus_update_max(batchSize, iC, iH, iW, (float*)max_frame, (float*)to_compare, (int32_t*)argmax_frame, conv_idx, num_els_per_image, stream);
    case DataType::kHALF:
        return inference_horus_update_max(batchSize, iC, iH, iW, (__half*)max_frame, (__half*)to_compare, (int32_t*)argmax_frame, conv_idx, num_els_per_image, stream);
    case DataType::kINT8:
    case DataType::kINT32:
    case DataType::kBOOL:
        break;
    }
    return 1;
}