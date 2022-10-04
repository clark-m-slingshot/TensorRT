#include "horusUpdateMaxPlugin.h"
#include <numeric>
#include <stdexcept>
#include <cuda_fp16.h>
#include <chrono>
#include <thread>


using namespace nvinfer1;
using nvinfer1::plugin::HorusUpdateMaxPlugin;
using nvinfer1::plugin::HorusUpdateMaxPluginCreator;

namespace
{
constexpr const char* HORUS_UPDATE_MAX_PLUGIN_VERSION{"1"};
constexpr const char* HORUS_UPDATE_MAX_PLUGIN_NAME{"HorusUpdateMax"};
} // namespace

// // Static class fields initialization
PluginFieldCollection HorusUpdateMaxPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> HorusUpdateMaxPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(HorusUpdateMaxPluginCreator);

HorusUpdateMaxPlugin::HorusUpdateMaxPlugin(int32_t convIdx)
    : mConvIdx(convIdx)
{
    PLUGIN_VALIDATE(convIdx > 0);
}

int HorusUpdateMaxPlugin::initialize() noexcept
{
    return 0;
}

HorusUpdateMaxPlugin::HorusUpdateMaxPlugin(const void* data, size_t length)
{
    // Deserialize in the same order as serialization
    deserialize_value(&data, &length, &mConvIdx);
}

const char* HorusUpdateMaxPlugin::getPluginType() const noexcept
{
    return HORUS_UPDATE_MAX_PLUGIN_NAME;
}

const char* HorusUpdateMaxPlugin::getPluginVersion() const noexcept
{
    return HORUS_UPDATE_MAX_PLUGIN_VERSION;
}

int HorusUpdateMaxPlugin::getNbOutputs() const noexcept
{
    return 2;
}

nvinfer1::DimsExprs HorusUpdateMaxPlugin::getOutputDimensions(
    int index, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    PLUGIN_ASSERT(nbInputs == 3);
    PLUGIN_ASSERT(index == 0 || index == 1);
    nvinfer1::DimsExprs outputDimsExprs(inputs[0]);
    return outputDimsExprs;
}

void HorusUpdateMaxPlugin::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept
{
}

// Detach the plugin object from its execution context.
void HorusUpdateMaxPlugin::detachFromContext() noexcept
{
}

int HorusUpdateMaxPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    std::cout << "enqueue" << std::endl;

    // Get the input dimensions
    nvinfer1::Dims inputDims = inputDesc[0].dims;
    std::cout << "inputDims.nbDims: " << inputDims.nbDims << std::endl;
    int iC = inputDims.d[0];
    int iH = inputDims.d[1];
    int iW = inputDims.d[2];
    std::cout << "iC, iH, IW: " << iC << ", " << iH << ", " << iW << std::endl;

    // void* input_cpu = malloc(iC*iH*iW*sizeof(float));
    // if(cudaSuccess == cudaMemcpy(input_cpu, inputs[0], iC*iH*iW*sizeof(float), cudaMemcpyDeviceToHost))
    // {
    //     float* in_cpu_f = (float*)input_cpu;
    //     for(int chanIdx = 0; chanIdx < iC; chanIdx++)
    //     {
    //         for(int rowIdx = 0; rowIdx < iH; rowIdx++)
    //         {
    //             for(int colIdx = 0; colIdx < iW; colIdx++)
    //             {
    //                 std::cout << in_cpu_f[rowIdx*iW + colIdx] << "\t"; 
    //             }
    //             std::cout << std::endl;
    //         }
    //     }
    // }

    // std::this_thread::sleep_for(std::chrono::seconds(10));

    // cuda memcpy into output buffers
    if(cudaSuccess == cudaMemcpy(outputs[1], inputs[2], iC*iH*iW*sizeof(int32_t), cudaMemcpyHostToDevice))
    {
        std::cout << "Copied argmax input into argmax output" << std::endl;
    }
    int32_t* argmaxFrameOut = static_cast<int32_t*>(outputs[1]);

    if(inputDesc->type == nvinfer1::DataType::kFLOAT)
    {
        if(cudaSuccess == cudaMemcpy(outputs[0], inputs[0], iC*iH*iW*sizeof(float), cudaMemcpyHostToDevice))
        {
            std::cout << "Copied max frame input into max frame output" << std::endl;
        }
        float* maxFrameOut = static_cast<float*>(outputs[0]);
        return horusUpdateMaxKernelLauncher(maxFrameOut, static_cast<const float*>(inputs[1]), argmaxFrameOut, mConvIdx, iC, iH, iW, stream);
    }
    else if(inputDesc->type == nvinfer1::DataType::kHALF)
    {
        if(cudaSuccess == cudaMemcpy(outputs[0], inputs[0], iC*iH*iW*sizeof(__half), cudaMemcpyHostToDevice))
        {
            std::cout << "Copied max frame input into max frame output" << std::endl;
        }
        __half* maxFrameOut = static_cast<__half*>(outputs[0]);
        return horusUpdateMaxKernelLauncher(maxFrameOut, static_cast<const __half*>(inputs[1]), argmaxFrameOut, mConvIdx, iC, iH, iW, stream);
    }
}

size_t HorusUpdateMaxPlugin::getSerializationSize() const noexcept
{
    return sizeof(mConvIdx);
}

void HorusUpdateMaxPlugin::serialize(void* buffer) const noexcept
{
    serialize_value(&buffer, mConvIdx);
}

bool HorusUpdateMaxPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    PLUGIN_ASSERT(inOut && pos < (nbInputs + nbOutputs));
    
    switch (pos)
    {
        case 0: return ((inOut[pos].type == nvinfer1::DataType::kFLOAT) || (inOut[pos].type == nvinfer1::DataType::kHALF));
        case 1: return ((inOut[pos].type == nvinfer1::DataType::kFLOAT) || (inOut[pos].type == nvinfer1::DataType::kHALF));
        case 2: return (inOut[pos].type == nvinfer1::DataType::kINT32);
        case 3: return (inOut[pos].type == inOut[0].type);
        case 4: return (inOut[pos].type == inOut[2].type);
    }
}

void HorusUpdateMaxPlugin::terminate() noexcept
{
}

void HorusUpdateMaxPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

IPluginV2DynamicExt* HorusUpdateMaxPlugin::clone() const noexcept
{
    try
    {
        auto* plugin = new HorusUpdateMaxPlugin(mConvIdx);
        plugin->setPluginNamespace(mPluginNamespace);
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void HorusUpdateMaxPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
}

nvinfer1::DataType HorusUpdateMaxPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    PLUGIN_ASSERT(inputTypes && nbInputs > 0);
    PLUGIN_ASSERT(index == 0 || index == 1);
    PLUGIN_ASSERT(inputTypes[2] == nvinfer1::DataType::kINT32);
    switch(index)
    {
        case 0: return inputTypes[0];
        case 1: return inputTypes[2];
    }
}

size_t HorusUpdateMaxPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}

void HorusUpdateMaxPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mPluginNamespace = libNamespace;
}

const char* HorusUpdateMaxPlugin::getPluginNamespace() const noexcept
{
    return mPluginNamespace;
}

HorusUpdateMaxPluginCreator::HorusUpdateMaxPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("conv_idx", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* HorusUpdateMaxPluginCreator::getPluginName() const noexcept
{
    return HORUS_UPDATE_MAX_PLUGIN_NAME;
}

const char* HorusUpdateMaxPluginCreator::getPluginVersion() const noexcept
{
    return HORUS_UPDATE_MAX_PLUGIN_VERSION;
}

const PluginFieldCollection* HorusUpdateMaxPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

const char* HorusUpdateMaxPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void HorusUpdateMaxPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

IPluginV2DynamicExt* HorusUpdateMaxPluginCreator::createPlugin(
    const char* name, const PluginFieldCollection* fc) noexcept
{
    try
    {
        int32_t convIdx{-1};
        for(int i = 0; i < fc->nbFields; i++)
        {
            std::string field_name(fc->fields[i].name);
            if(field_name.compare("conv_idx") == 0)
            {
                convIdx = *static_cast<const int32_t*>(fc->fields[i].data);
            }
        }

        HorusUpdateMaxPlugin* plugin = new HorusUpdateMaxPlugin(convIdx);
        plugin->setPluginNamespace(mNamespace.c_str());

        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2DynamicExt* HorusUpdateMaxPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    try
    {
        HorusUpdateMaxPlugin* plugin = new HorusUpdateMaxPlugin(serialData, serialLength);
        plugin->setPluginNamespace(mNamespace.c_str());

        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}