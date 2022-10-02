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

#include "horus_update_max_trt_plugin.h"
#include <cstring>
#include <iostream>
#include <vector>

using namespace nvinfer1;
// const int NUM_COORDCONV_CHANNELS = 2;

namespace
{
const char* HORUS_UPDATE_MAX_PLUGIN_VERSION{"1"};
const char* HORUS_UPDATE_MAX_PLUGIN_NAME{"HorusUpdateMax"};
} // namespace

PluginFieldCollection horus_update_max_plugin_creator::mFC{};
std::vector<PluginField> horus_update_max_plugin_creator::mPluginAttributes;

horus_update_max_trt_plugin::horus_update_max_trt_plugin() {}

horus_update_max_trt_plugin::horus_update_max_trt_plugin(nvinfer1::DataType iType, int conv_idx, int iC, int iH, int iW)
    : iType(iType)
    , conv_idx(conv_idx)
    , iC(iC)
    , iH(iH)
    , iW(iW)
{
}

horus_update_max_trt_plugin::horus_update_max_trt_plugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    iType = read<nvinfer1::DataType>(d);
    conv_idx = read<int>(d);
    num_els = read<int>(d);
    PLUGIN_VALIDATE(d == a + length);
}

int horus_update_max_trt_plugin::getNbOutputs() const noexcept
{
    return 2;
}

int horus_update_max_trt_plugin::initialize() noexcept
{
    return STATUS_SUCCESS;
}

void horus_update_max_trt_plugin::terminate() noexcept {}

Dims horus_update_max_trt_plugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept
{
    // CHW
    nvinfer1::Dims dimsOutput;
    dimsOutput.nbDims = inputs->nbDims;
    dimsOutput.d[0] = inputs->d[0];
    dimsOutput.d[1] = inputs->d[1];
    dimsOutput.d[2] = inputs->d[2];
    dimsOutput.d[3] = inputs->d[3];
    return dimsOutput;
}

size_t horus_update_max_trt_plugin::getWorkspaceSize(int maxBatchSize) const noexcept
{
    return 0;
}

size_t horus_update_max_trt_plugin::getSerializationSize() const noexcept
{
    // iType, conv_idx, iC, iH, iW
    return sizeof(nvinfer1::DataType) + sizeof(int) * 4;
}

void horus_update_max_trt_plugin::serialize(void* buffer) const noexcept
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, iType);
    write(d, conv_idx);
    write(d, iC);
    write(d, iH);
    write(d, iW);
    PLUGIN_ASSERT(d == a + getSerializationSize());
}

void horus_update_max_trt_plugin::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, nvinfer1::PluginFormat format, int maxBatchSize) noexcept
{
    PLUGIN_ASSERT(nbInputs == 2);
    PLUGIN_ASSERT(nbOutputs == 2);

    iC = inputDims->d[0];
    iH = inputDims->d[1];
    iW = inputDims->d[2];

    iType = inputTypes[0];
}

bool horus_update_max_trt_plugin::supportsFormat(DataType type, PluginFormat format) const noexcept
{
    // kLinear specifies format of tensors. https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/namespacenvinfer1.html#ac3e115b1a2b1e578e8221ef99d27cd45
    return ((type == DataType::kFLOAT || type == DataType::kHALF) && format == PluginFormat::kLINEAR);
}

const char* horus_update_max_trt_plugin::getPluginType() const noexcept
{
    return HORUS_UPDATE_MAX_PLUGIN_NAME;
}

const char* horus_update_max_trt_plugin::getPluginVersion() const noexcept
{
    return HORUS_UPDATE_MAX_PLUGIN_VERSION;
}

void horus_update_max_trt_plugin::destroy() noexcept
{
    delete this;
}

IPluginV2Ext* horus_update_max_trt_plugin::clone() const noexcept
{
    try
    {
        auto* plugin = new horus_update_max_trt_plugin(iType, conv_idx, iC, iH, iW);
        plugin->setPluginNamespace(mPluginNamespace);
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void horus_update_max_trt_plugin::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mPluginNamespace = pluginNamespace;
}

const char* horus_update_max_trt_plugin::getPluginNamespace() const noexcept
{
    return mPluginNamespace;
}

nvinfer1::DataType horus_update_max_trt_plugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    return inputTypes[0];
}

bool horus_update_max_trt_plugin::isOutputBroadcastAcrossBatch(
    int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept
{
    return false;
}

bool horus_update_max_trt_plugin::canBroadcastInputAcrossBatch(int inputIndex) const noexcept
{
    return false;
}

// Plugin creator
horus_update_max_plugin_creator::horus_update_max_plugin_creator() {}

const char* horus_update_max_plugin_creator::getPluginName() const noexcept
{
    return HORUS_UPDATE_MAX_PLUGIN_NAME;
}

const char* horus_update_max_plugin_creator::getPluginVersion() const noexcept
{
    return HORUS_UPDATE_MAX_PLUGIN_VERSION;
}

const PluginFieldCollection* horus_update_max_plugin_creator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2Ext* horus_update_max_plugin_creator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    try
    {
        horus_update_max_trt_plugin* plugin = new horus_update_max_trt_plugin();
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2Ext* horus_update_max_plugin_creator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    try
    {
        horus_update_max_trt_plugin* plugin = new horus_update_max_trt_plugin(serialData, serialLength);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}