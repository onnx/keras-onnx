# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################

"""
    Define the common constant data and type for the converter.
"""


class TYPES:
    # tf-node types:
    Identity = 'Identity'
    Const = 'Const'
    AddN = 'AddN'
    Any = 'Any'
    All = 'All'
    ArgMax = 'ArgMax'
    ArgMin = 'ArgMin'
    AvgPool = 'AvgPool'
    AvgPool3D = 'AvgPool3D'
    BatchMatMul = 'BatchMatMul'
    BatchMatMulV2 = 'BatchMatMulV2'
    BatchToSpaceND = 'BatchToSpaceND'
    BiasAdd = 'BiasAdd'
    BiasAddV1 = 'BiasAddV1'
    Cast = 'Cast'
    ConcatV2 = 'ConcatV2'
    Conv1D = 'Conv1D'
    Conv2D = 'Conv2D'
    Conv3D = 'Conv3D'
    Cumsum = 'Cumsum'
    DepthToSpace = 'DepthToSpace'
    DepthwiseConv2dNative = 'DepthwiseConv2dNative'
    Div = 'Div'
    Einsum = 'Einsum'
    ExpandDims = 'ExpandDims'
    Fill = 'Fill'
    FloorDiv = 'FloorDiv'
    FloorMod = 'FloorMod'
    FusedBatchNorm = 'FusedBatchNorm'
    FusedBatchNormV2 = 'FusedBatchNormV2'
    FusedBatchNormV3 = 'FusedBatchNormV3'
    GatherNd = 'GatherNd'
    GatherV2 = 'GatherV2'
    GreaterEqual = 'GreaterEqual'
    LessEqual = 'LessEqual'
    LogicalAnd = 'LogicalAnd'
    LogicalNot = 'LogicalNot'
    LogSoftmax = 'LogSoftmax'
    MatMul = 'MatMul'
    Max = 'Max'
    Maximum = 'Maximum'
    MaxPool = 'MaxPool'
    MaxPoolV2 = 'MaxPoolV2'
    MaxPool3D = 'MaxPool3D'
    Mean = 'Mean'
    Min = 'Min'
    Minimum = 'Minimum'
    NonMaxSuppressionV2 = 'NonMaxSuppressionV2'
    NonMaxSuppressionV3 = 'NonMaxSuppressionV3'
    NotEqual = 'NotEqual'
    OneHot = 'OneHot'
    Pack = 'Pack'
    Pad = 'Pad'
    PadV2 = 'PadV2'
    Pow = 'Pow'
    Prod = 'Prod'
    Range = 'Range'
    ReadVariableOp = 'ReadVariableOp'
    RealDiv = 'RealDiv'
    Relu6 = 'Relu6'
    Reshape = 'Reshape'
    ResizeBilinear = 'ResizeBilinear'
    ResizeNearestNeighbor = 'ResizeNearestNeighbor'
    ReverseV2 = 'ReverseV2'
    Round = 'Round'
    Rsqrt = 'Rsqrt'
    ScatterNd = 'ScatterNd'
    Select = 'Select'
    Shape = 'Shape'
    Size = 'Size'
    Slice = 'Slice'
    Softmax = 'Softmax'
    SpaceToBatchND = 'SpaceToBatchND'
    Split = 'Split'
    SplitV = 'SplitV'
    Square = 'Square'
    SquaredDifference = 'SquaredDifference'
    Squeeze = 'Squeeze'
    StridedSlice = 'StridedSlice'
    Sum = 'Sum'
    Tile = 'Tile'
    TopKV2 = 'TopKV2'
    Transpose = 'Transpose'
    Unpack = 'Unpack'
    VarHandleOp = 'VarHandleOp'
    VariableV2 = 'VariableV2'
    Where = 'Where'
    ZerosLike = 'ZerosLike'

    # converter internal types:
    TD_Reshape = '_reshape_timedistributed'


NCHW_TO_NHWC = [0, 2, 3, 1]
NHWC_TO_NCHW = [0, 3, 1, 2]
HWCN_TO_NCHW = [3, 2, 0, 1]
NCHW_TO_HWCN = [2, 3, 1, 0]
NCDHW_TO_NDHWC = [0, 2, 3, 4, 1]
NDHWC_TO_NCDHW = [0, 4, 1, 2, 3]
DHWCN_TO_NCDHW = [4, 3, 0, 1, 2]
