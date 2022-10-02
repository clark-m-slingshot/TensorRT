
#ifndef TRT_HORUS_UPDATE_MAX_PLUGIN_H
#define TRT_HORUS_UPDATE_MAX_PLUGIN_H

#include "NvInferPlugin.h"
#include "kernel.h"
#include "plugin.h"
#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace nvinfer1
{
namespace plugin
{

class horus_update_max_trt_plugin : public IPluginV2Ext
{
public:
    horus_update_max_trt_plugin();

    horus_update_max_trt_plugin(DataType iType, int conv_idx, int iC, int iH, int iW);
    // horus_update_max_trt_plugin(DataType iType, int iC, int iH, int iW, int oC, int oH, int oW);

    horus_update_max_trt_plugin(const void* data, size_t length);

    ~horus_update_max_trt_plugin() override = default;

    int getNbOutputs() const noexcept override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept override;

    int initialize() noexcept override;

    void terminate() noexcept override;

    size_t getWorkspaceSize(int maxBatchSize) const noexcept override;

    int enqueue(int batchSize, void* const* max_frame, const void* const* to_compare, void* const* argmax_frame,
                void* workspace, cudaStream_t stream) noexcept override;

    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

    void configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
        const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
        const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) noexcept override;

    bool supportsFormat(DataType type, PluginFormat format) const noexcept override;

    const char* getPluginType() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    void destroy() noexcept override;

    IPluginV2Ext* clone() const noexcept override;

    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputType, int nbInputs) const noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept override;

    bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override;

private:
    DataType iType;
    int conv_idx, iC, iH, iW;
    const char* mPluginNamespace;
};

class horus_update_max_plugin_creator : public nvinfer1::pluginInternal::BaseCreator
{
public:
    horus_update_max_plugin_creator();

    ~horus_update_max_plugin_creator() override = default;

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    const PluginFieldCollection* getFieldNames() noexcept override;

    IPluginV2Ext* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;

    IPluginV2Ext* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;

protected:
    std::string mNamespace;
};
} // namespace plugin
} // namespace nvinfer1

#endif // TRT_HORUS_UPDATE_MAX_PLUGIN_H