#ifndef TRT_HORUS_UPDATE_MAX_PLUGIN_H
#define TRT_HORUS_UPDATE_MAX_PLUGIN_H

#include "common/plugin.h"
#include "common/serialize.hpp"
#include <cudnn.h>
#include <iostream>
#include <string>
#include <vector>

// One of the preferred ways of making TensorRT to be able to see
// our custom layer requires extending IPluginV2 and IPluginCreator classes.
// For requirements for overriden functions, check TensorRT API docs.
namespace nvinfer1
{
namespace plugin
{

template <typename T>
cudaError_t horusUpdateMaxKernelLauncher(const T* maxFrameIn, const T* toCompare, const int32_t* argmaxFrameIn, const int32_t convIdx, int iC, int iH, int iW, T* maxFrameOut, int32_t* argmaxFrameOut, cudaStream_t stream);

class HorusUpdateMaxPlugin final : public nvinfer1::IPluginV2DynamicExt
{
public:
    HorusUpdateMaxPlugin(int32_t conv_idx);

    HorusUpdateMaxPlugin() = delete;

    HorusUpdateMaxPlugin(const void* data, size_t length);

    int getNbOutputs() const noexcept override;

    // DynamicExt plugins returns DimsExprs class instead of Dims
    DimsExprs getOutputDimensions(int index, const nvinfer1::DimsExprs* inputs, int nbInputDims,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    int initialize() noexcept override;

    void terminate() noexcept override;

    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override;

    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

    bool supportsFormatCombination(
        int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;

    const char* getPluginType() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;

    void destroy() noexcept override;

    DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;

    void attachToContext(cudnnContext* cudnn, cublasContext* cublas, nvinfer1::IGpuAllocator* allocator) noexcept override;

    void detachFromContext() noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;

private:
    const char* mPluginNamespace;
    std::string mNamespace;
    int32_t mConvIdx;
};

class HorusUpdateMaxPluginCreator : public IPluginCreator
{
public:
    HorusUpdateMaxPluginCreator();

    ~HorusUpdateMaxPluginCreator() override = default;

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    const PluginFieldCollection* getFieldNames() noexcept override;

    IPluginV2DynamicExt* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;

    IPluginV2DynamicExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;
};
} // namespace plugin
} // namespace nvinfer1

#endif // TRT_HORUS_UPDATE_MAX_PLUGIN_H