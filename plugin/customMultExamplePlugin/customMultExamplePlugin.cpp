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
#include <numeric>
#include <stdexcept>

using namespace nvinfer1;
using nvinfer1::plugin::CustomMultExamplePlugin;
using nvinfer1::plugin::CustomMultExamplePluginCreator;

namespace
{
constexpr const char* CUSTOM_MULT_EXAMPLE_PLUGIN_VERSION{"1"};
constexpr const char* CUSTOM_MULT_EXAMPLE_PLUGIN_NAME{"CustomMult"};
} // namespace

// // Static class fields initialization
PluginFieldCollection CustomMultExamplePluginCreator::mFC{};
std::vector<nvinfer1::PluginField> CustomMultExamplePluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(CustomMultExamplePluginCreator);

CustomMultExamplePlugin::CustomMultExamplePlugin(int32_t conv_idx)
    : mConvIdx(conv_idx)
{
    PLUGIN_VALIDATE(conv_idx > 0);
}

int CustomMultExamplePlugin::initialize() noexcept
{
    return 0;
}

CustomMultExamplePlugin::CustomMultExamplePlugin(const void* data, size_t length)
{
    // Deserialize in the same order as serialization
    deserialize_value(&data, &length, &mConvIdx);
}

const char* CustomMultExamplePlugin::getPluginType() const noexcept
{
    return CUSTOM_MULT_EXAMPLE_PLUGIN_NAME;
}

const char* CustomMultExamplePlugin::getPluginVersion() const noexcept
{
    return CUSTOM_MULT_EXAMPLE_PLUGIN_VERSION;
}

int CustomMultExamplePlugin::getNbOutputs() const noexcept
{
    return 1;
}

nvinfer1::DimsExprs CustomMultExamplePlugin::getOutputDimensions(
    int index, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    PLUGIN_ASSERT(nbInputs == 1);
    PLUGIN_ASSERT(index == 0);
    std::cout << inputs[0].d[0]->getConstantValue() << std::endl;
    std::cout << inputs[0].d[1]->getConstantValue() << std::endl;
    std::cout << inputs[0].d[2]->getConstantValue() << std::endl;
    std::cout << inputs[0].d[3]->getConstantValue() << std::endl;
    std::cout << inputs[1].d[0] << std::endl;
    nvinfer1::DimsExprs output(inputs[0]);
    return output;
}

void CustomMultExamplePlugin::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept
{
}

// Detach the plugin object from its execution context.
void CustomMultExamplePlugin::detachFromContext() noexcept
{
}

int CustomMultExamplePlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    std::cout << "here" << std::endl;
    // Get the input dimensions
    nvinfer1::Dims input_dims = inputDesc[0].dims;
    int batchSize = input_dims.d[0];
    int nbChannels = input_dims.d[1];
    int height = input_dims.d[2];
    int width = input_dims.d[3];
    std::cout << "batchSize: " << batchSize << std::endl;
    std::cout << "nbChannels: " << nbChannels << std::endl;
    std::cout << "height: " << height << std::endl;
    std::cout << "width: " << width << std::endl;

    float* output = (float*)inputs[0];
    return customMultExamplePluginInference(output, nbChannels, height, width, mConvIdx, stream);
}

size_t CustomMultExamplePlugin::getSerializationSize() const noexcept
{
    return sizeof(mConvIdx);
}

void CustomMultExamplePlugin::serialize(void* buffer) const noexcept
{
    serialize_value(&buffer, mConvIdx);
}

bool CustomMultExamplePlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    PLUGIN_ASSERT(inOut && pos < (nbInputs + nbOutputs));

    std::cout << "nmInputs: " << nbInputs << ", nbOutputs: " << nbOutputs << std::endl;
    
    switch (pos)
    {
        case 0: std::cout << "About to return " << (inOut[pos].type == nvinfer1::DataType::kFLOAT) << std::endl; return inOut[pos].type == nvinfer1::DataType::kFLOAT;
        case 1: std::cout << "About to return " << (inOut[pos].type == nvinfer1::DataType::kFLOAT) << std::endl; return inOut[pos].type == nvinfer1::DataType::kFLOAT;
    }
}

void CustomMultExamplePlugin::terminate() noexcept
{
}

void CustomMultExamplePlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

IPluginV2DynamicExt* CustomMultExamplePlugin::clone() const noexcept
{
    try
    {
        auto* plugin = new CustomMultExamplePlugin(mConvIdx);
        plugin->setPluginNamespace(mPluginNamespace);
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void CustomMultExamplePlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
}

nvinfer1::DataType CustomMultExamplePlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    PLUGIN_ASSERT(inputTypes && nbInputs > 0 && index == 0);
    std::cout << "Returning output data type as type " << (int32_t)inputTypes[0] << std::endl;
    return inputTypes[0];
}

size_t CustomMultExamplePlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}

void CustomMultExamplePlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mPluginNamespace = libNamespace;
}

const char* CustomMultExamplePlugin::getPluginNamespace() const noexcept
{
    return mPluginNamespace;
}

CustomMultExamplePluginCreator::CustomMultExamplePluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("conv_idx", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* CustomMultExamplePluginCreator::getPluginName() const noexcept
{
    return CUSTOM_MULT_EXAMPLE_PLUGIN_NAME;
}

const char* CustomMultExamplePluginCreator::getPluginVersion() const noexcept
{
    return CUSTOM_MULT_EXAMPLE_PLUGIN_VERSION;
}

const PluginFieldCollection* CustomMultExamplePluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

const char* CustomMultExamplePluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void CustomMultExamplePluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

IPluginV2DynamicExt* CustomMultExamplePluginCreator::createPlugin(
    const char* name, const PluginFieldCollection* fc) noexcept
{
    try
    {
        int32_t conv_idx{-1};
        for(int i = 0; i < fc->nbFields; i++)
        {
            std::string field_name(fc->fields[i].name);
            if(field_name.compare("conv_idx") == 0)
            {
                conv_idx = *static_cast<const int32_t*>(fc->fields[i].data);
                std::cout << "Extracted " << conv_idx << " from attr" << std::endl;
            }
        }

        CustomMultExamplePlugin* plugin = new CustomMultExamplePlugin(conv_idx);
        plugin->setPluginNamespace(mNamespace.c_str());

        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2DynamicExt* CustomMultExamplePluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    try
    {
        CustomMultExamplePlugin* plugin = new CustomMultExamplePlugin(serialData, serialLength);
        plugin->setPluginNamespace(mNamespace.c_str());

        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}