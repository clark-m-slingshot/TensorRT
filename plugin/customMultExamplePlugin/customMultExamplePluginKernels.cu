// #include "customMultExamplePlugin.h"
// #include <cuda_fp16.h>

const int num_threads_per_block = 1024; // max threads per block for compute capability

/**
 * Computes optimal number of thread blocks (1 thread per array element)
 *
 * @param   total_els   total number of elements to process; i.e., total number of threads to launch
 * @return  num_thread_blocks   the number of threads blocks to use in order to accomodate number of needed threads
 */
int get_num_thread_blocks_trt_custMult(int total_els)
{
    int num_thread_blocks = total_els/num_threads_per_block;
    if(total_els % num_threads_per_block != 0)
    {
        num_thread_blocks+=1;
    }

    return num_thread_blocks;
}

// template <typename T_DATA>
// __global__ void kernelCustomMult(int num_els, int multiplier, T_DATA* input)
// {
//     int index = blockIdx.x * blockDim.x + threadIdx.x;
//     if (index < num_els)
//     {
//         input[index] *= multiplier;
//     }
//     __syncthreads();
// }

// template <typename T>
// int inferenceCustomMult(
//     int batchSize, int iC, int iH, int iW, T* inputs, int* multiplier, cudaStream_t stream)
// {
//     int num_els_per_tens = iC*iH*iW;
//     int num_thread_blocks = get_num_thread_blocks_trt_custMult(num_els_per_tens);

//     for (int i = 0; i < batchSize; ++i)
//     {
//         // NOTE: kernelCopy kernel can be replaced with cudaMemcpy function
//         kernelCustomMult<<<num_thread_blocks, num_threads_per_block, 0, stream>>>(num_els_per_tens, *multiplier, inputs);
//         inputs += iC*iH*iW;
//     }

//     cudaError_t err = cudaGetLastError();
//     if (cudaSuccess != err)
//     {
//         fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
//         return 1;
//     }
//     return 0;
// }

// int32_t CustomMultExamplePlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
//     nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workSpace,
//     cudaStream_t stream) PLUGIN_NOEXCEPT
// {
//     int32_t const batch = inputDesc[0].dims.d[0];
//     int32_t iC = inputDesc[0].dims.d[1];
//     int32_t iH = inputDesc[0].dims.d[2];
//     int32_t iW = inputDesc[0].dims.d[3];
//     switch (iType)
//     {
//     case DataType::kFLOAT:
//         return inferenceCustomMult(batchSize, iC, iH, iW, (float*) inputs[0], (int*) inputs[1], stream);
//     case DataType::kHALF:
//         return inferenceCustomMult(batchSize, iC, iH, iW, (__half*) inputs[0], (int*) inputs[1], stream);
//     case DataType::kINT8:
//     case DataType::kINT32:
//     case DataType::kBOOL:
//         break;
//     }
//     return 1;
// }

/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "customMultExamplePlugin.h"

namespace nvinfer1
{
namespace plugin
{

template <typename T>
__global__ void customMultExamplePluginKernel(int num_els, int multiplier, T* input)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_els)
    {
        input[index] *= multiplier;
    }
}

// template <typename T, unsigned TPB>
// __global__ void scaleShiftChannelsInplaceKernel(T* inOut, const int ld, const float* beta, const float* gamma)
// {
//     // grid is blocks x C x B
//     // ld should be H*W
//     // blockIdx.z = batch
//     // blockIdx.y = channel
//     // blockIdx.x = block per col
//     const T b = beta[blockIdx.y];
//     const T g = gamma[blockIdx.y];

//     const int offset = (blockIdx.z * gridDim.y + blockIdx.y) * ld;

//     const int tx = blockIdx.x * TPB + threadIdx.x;

//     if (tx < ld)
//     {
//         inOut[offset + tx] = g * inOut[offset + tx] + b;
//     }
// }

template <typename T>
cudaError_t customMultExamplePluginInference(T* input, int iC, int iH, int iW, const int32_t multiplier, cudaStream_t stream)
{
    int num_els_per_tens = iC*iH*iW;
    int num_thread_blocks = get_num_thread_blocks_trt_custMult(num_els_per_tens);

    customMultExamplePluginKernel<<<num_thread_blocks, num_threads_per_block, 0, stream>>>(num_els_per_tens, multiplier, input);

    return cudaPeekAtLastError();
}

// template <typename T>
// cudaError_t scaleShiftChannelsInplace(T* inOut, const int B, const int C, const int channelVolume, const float* beta,
//     const float* gamma, cudaStream_t stream)
// {

//     constexpr int TPB = 256;
//     const int colBlocks = (channelVolume + TPB - 1) / TPB;
//     const dim3 grid(colBlocks, C, B);

//     scaleShiftChannelsInplaceKernel<T, TPB><<<grid, TPB, 0, stream>>>(inOut, channelVolume, beta, gamma);

//     return cudaPeekAtLastError();
// }

template cudaError_t customMultExamplePluginInference<float>(float* input, int iC, int iH, int iW, const int32_t multiplier, cudaStream_t stream);

// template cudaError_t scaleShiftChannelsInplace<float>(float* inOut, const int B, const int C, const int channelVolume, const float* beta,
//     const float* gamma, cudaStream_t stream);

} /* plugin */
} /* nvinfer1 */