import torch
from torch import nn
import numpy as np
from typing import Union, Type, List, Tuple
import logging

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op
from nnunetv2.network_architecture.layers import ReLUConvDropOutNormMax, InputNormConvNorm
from nnunetv2.network_architecture.helper import get_matching_unpooling

# Default logging for debugging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


class MultiInputOutputSequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class CNNEncoder(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 return_skips: bool = False,
                 nonlin_first: bool = False,
                 pool: str = 'conv',
                 return_pool_idx: bool = True,
                 ):
        """
        Args:
            input_channels: number of input channels (i.e. 3 for RGB, 1 for Gray scale, 7 for thick slices)
            n_stages: number of resolution stages (~downsampling operations/number of blocks, excluding bottleneck)
            features_per_stage: number of filters per stage (i.e. 64 for all, or increasing 16-32-64-...)
            conv_op: name of convolution operation (i.e. nn.Conv1d, nn.Conv2d, nn.Conv3d)
            kernel_sizes: Kernel-filter size (i.e. 3x3, 5x5, 1x1)
            strides: List determining pool and convolution strides; should be 2 for MaxPool in FCNN, 1 for Conv op
            n_conv_per_stage: number of convolutions per stage; 4 for CDB (3x3 -> 5x5 -> 7x7 -> 9x9 RF),
            conv_bias: If True, adds a learnable bias to the output of the convolution (for nn.Conv1d/2d/3d)
            norm_op: Normalization operation (i.e. BatchNorm - nn.BatchNorm2d/3d, GroupNorm,...)
            norm_op_kwargs: Arguments for Normalization Operation
                            (e.g. num_features, eps, momentum, affine, track_running_stats)
            dropout_op: Dropout operation (i.e. nn.DropOut)
            dropout_op_kwargs: Arguments for Drop out (p = probability of zeroing, inplace)
            nonlin: Non-linearity to use (i.e. ReLU, pReLU, leaky ReLU)
            nonlin_kwargs: Arguments for non-linearity
                           (e.g. negative_slope for leaky relu)
            return_skips: Return skip-connections
            nonlin_first: Apply the non-linearity first, then normalization
                          (swap order, normally Conv-Drop-Norm-NonLin --> swap = Conv-Drop-NonLin-Norm)
            pool: Pooling operation (Max, Avg, Conv)
            return_pool_idx: Return indices of max-pooling op (for unpooling with indices in decoder)

        """

        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        assert len(kernel_sizes) == n_stages, "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(features_per_stage) == n_stages, "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, "strides must have as many entries as we have resolution stages (n_stages). " \
                                         "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"

        stages = []
        pooling = []
        first = True
        for s in range(n_stages):
            logging.info(f"Stage {s}")
            if pool == 'max' or pool == 'avg':
                if (isinstance(strides[s], int) and strides[s] != 1) or \
                        isinstance(strides[s], (tuple, list)) and any([i != 1 for i in strides[s]]):
                    # kernel_size=params["pool"]=2, stride_params=["stride_pool"]=2 for FCNN
                    # --> get pool operation (maxPool2D or 3D)
                    # add pooling op, if stride is != 1, otherwise only add CDB to the stage_modules list
                    pooling.append(get_matching_pool_op(conv_op, pool_type=pool)(kernel_size=strides[s],
                                                                                       stride=strides[s],
                                                                                       return_indices=return_pool_idx))
                    logging.debug(f"Pool {pool} return {return_pool_idx} with stride {strides[s]}")
                conv_stride = 1
            elif pool == 'conv':
                conv_stride = strides[s]
            else:
                raise RuntimeError()
            # Add the CompetitiveDenseBloc --> for FCNN, the very first one should be different (inputCDB)
            logging.debug(f"Nr Conv ops {n_conv_per_stage[s]}, input C {input_channels}, features {features_per_stage[s]},"
                  f"kernel {kernel_sizes[s]}")
            stages.append(CompetitiveDenseBlocks(
                n_conv_per_stage[s], conv_op, input_channels, features_per_stage[s], kernel_sizes[s], conv_stride,
                first, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs
            ))

            first = False
            # Create Sequential list from stage_modules (careful: can not pass any parameters to nn.Sequential
            # in the forward call! --> might need own implementation for VINN here; also would need output from
            # Pooling and Convolution if pool_idx is defined!)
            # stages.append(MultiInputOutputSequential(*stage_modules))

            # Change input channels to the number of output features after the CDB block
            input_channels = features_per_stage[s]

        self.stages = nn.ModuleList(stages)
        self.pooling = nn.ModuleList(pooling)
        self.output_channels = features_per_stage  # list of features per block
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips
        self.return_pool_idx = return_pool_idx

        # we store some things that a potential decoder needs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x):
        """
        Forward call to FastSurferCNN - Competitive Encoder Block
        Generell set-up:
            Defines order of operations via strides arguments --> list with entries, entry=1 conv, entry=2 Pool
            Example for Classic FastsurferCNN would be:
            CDB (s1 output) ->
                [ Pool -> CDB ] (p1, s2 output) ->
                    [ Pool -> CDB ] (p2, s3 output) ->
                        [ Pool -> CDB ] (p3, s4 output) ->
                            [ Pool -> CDB ] (p4, s5 output = bottleneck) -> out
        Args:
            x: image (patch) to process

        Returns:
            if self.return_skips is False and self.return_pool_idx is False:
                1. Only the Bottleneck layer encoded feature map (=s5 output)
            if self.return_skips = True and self.return_pool_idx = False:
                2. The Encoded feature maps of each layer (s1, s2, s3, s4, s5)
            if self.return_skips = True and self.return_pool_idx = True:
                3. Encoded feature maps and indices for pooling operation ((s1), (s2, p2), (s3, p3), (s4, p4), (s5, p5))
        """
        ret = []
        pidx = []
        for s in range(len(self.stages)-1):
            # We start with the convolution, then the pooling operation
            x = self.stages[s](x)
            logging.debug(f"Block {s}, shape out {x.shape}")
            # Append convolution before pooling for the skip connections
            ret.append(x)
            # Now perform the pooling operation --> output here is input to next CDB Block
            x = self.pooling[s](x)
            if self.return_pool_idx:
                # If we have index unpooling, we get the index and the pooled feature map
                # Otherwise, only the pooled feature maps are returned
                pidx.append(x[1])
                x = x[0]

        # Final convolution in bottleneck block (does not have associated pooling op)
        x = self.stages[-1](x)
        logging.debug(f"Block fin, shape out {x.shape}")
        ret.append(x)  # Append last result (bottleneck block, no Pooling)
        # Return skip connections
        if not self.return_skips and not self.return_pool_idx:
            # Only return final output
            return x
        elif not self.return_pool_idx:
            # Return skip connections, but not pooling idx (because they are not defined)
            return ret
        else:
            return ret, pidx

    def compute_conv_feature_map_size(self, input_size):
        output = np.int64(0)
        for s in range(len(self.stages)):
            if isinstance(self.stages[s], nn.Sequential):
                for sq in self.stages[s]:
                    if hasattr(sq, 'compute_conv_feature_map_size'):
                        output += self.stages[s][-1].compute_conv_feature_map_size(input_size)
            else:
                output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]
        return output


class CNNDecoder(nn.Module):
    def __init__(self,
                 encoder: CNNEncoder,
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision, index_unpool: bool = False, nonlin_first: bool = False):
        """
        This class needs the skips of the encoder as input in its forward.
        the encoder goes all the way to the bottleneck, so that's where the decoder picks up. stages in the decoder
        are sorted by order of computation, so the first stage has the lowest resolution and takes the bottleneck
        features and the lowest skip as inputs
        the decoder has two (three) parts in each stage:
        1) conv transpose to upsample the feature maps of the stage below it (or the bottleneck in case of the first stage)
        2) n_conv_per_stage conv blocks to let the two inputs get to know each other and merge
        3) (optional if deep_supervision=True) a segmentation output Todo: enable upsample logits?
        :param encoder:
        :param num_classes:
        :param n_conv_per_stage:
        :param deep_supervision:
        """
        super().__init__()
        self.deep_supervision = deep_supervision
        self.index_unpool = index_unpool
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                          "resolution stages - 1 (n_stages in encoder - 1), " \
                                                          "here: %d" % n_stages_encoder

        transpconv_op = get_matching_unpooling(conv_op=encoder.conv_op, idx_unpool=index_unpool)

        # we start with the bottleneck and work out way up
        stages = []
        transpconvs = []
        seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            if not self.index_unpool:
                transpconvs.append(transpconv_op(input_features_below, input_features_skip, stride_for_transpconv,
                                                 stride_for_transpconv, bias=encoder.conv_bias
                                                 )
                                   )
            else:
                transpconvs.append(transpconv_op(kernel_size=stride_for_transpconv, stride=stride_for_transpconv))

            # input features to conv is 1 x input_features_skip (concat replaced with maximum for
            # input_features_skip with transpconv output)
            stages.append(CompetitiveDenseBlocks(
                n_conv_per_stage[s-1], encoder.conv_op, input_features_skip, input_features_skip,
                encoder.kernel_sizes[-(s + 1)], 1, False, encoder.conv_bias, encoder.norm_op, encoder.norm_op_kwargs,
                encoder.dropout_op, encoder.dropout_op_kwargs, encoder.nonlin, encoder.nonlin_kwargs
            ))

            # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
            # then a model trained with deep_supervision=True could not easily be loaded at inference time where
            # deep supervision is not needed. It's just a convenience thing
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))  # 1x1 convolutions

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips, *args):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        Forward call to FastSurferCNN - Competitive Decoder Block
        Generell set-up:
            Defines order of operations via strides arguments --> list with entries, entry=1 conv, entry=2 Pool
        Example for Classic FastsurferCNN would be:

        s1 ---------------------------------------------------------------> [ Transpose/Unpool -> CDB ] -> o4 -> 1x1 conv
                                                                                         ^
        p2, s2 ----------------------------------------> [ Transpose/Unpool -> CDB ] -> o3 ---> 1x1 conv
                                                                    ^
        p3, s3 ---------------------------> [ Transpose/Unpool -> CDB ] -> o2 ---> 1x1 conv
                                                        ^
        p4, s4      --> [ Transpose/Unpool -> CDB ] -> o1 ---> 1x1 conv (if deep_supervision, otherwise only last layer)
            s5 = skips[-1] ^
        Args:
            skips: outputs of encoder (skip connections)
            *args: optional, index for unpooling op, if index unpooling should be used and maxpooling was used in encoder
        """
        lres_input = skips[-1]
        seg_outputs = []
        logging.debug(len(args[0]), len(skips))
        if args:
            indices = args[0]
        for s in range(len(self.stages)):
            logging.debug(f"Stage {s}, idx: {-(s+1)}, skip: {-(s+2)}")
            if self.index_unpool:
                x = self.transpconvs[s](lres_input, indices[-(s+1)])
            else:
                x = self.transpconvs[s](lres_input)
            logging.debug(f"Orig shape: {lres_input.shape}, Upsample shape: {x.shape}, Skip shape: {skips[-(s+2)].shape}")
            x = torch.maximum(x, skips[-(s+2)])
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

    def compute_conv_feature_map_size(self, input_size):
        """
        IMPORTANT: input_size is the input_size of the encoder!
        :param input_size:
        :return:
        """
        # first we need to compute the skip sizes. Skip bottleneck because all output feature maps of our ops will at
        # least have the size of the skip above that (therefore -1)
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]
        # logging.debug(skip_sizes)

        assert len(skip_sizes) == len(self.stages)

        # our ops are the other way around, so let's match things up
        output = np.int64(0)
        for s in range(len(self.stages)):
            # logging.debug(f"stage {s}, Skip sizes: {skip_sizes[-(s+1)]}, output_channels: {self.encoder.output_channels[-(s+2)]}")
            # conv blocks
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s+1)])
            # trans conv
            if not self.index_unpool:
                output += np.prod([self.encoder.output_channels[-(s+2)], *skip_sizes[-(s+1)]], dtype=np.int64)
            # segmentation
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s+1)]], dtype=np.int64)
        return output


class CompetitiveDenseBlocks(nn.Module):
    def __init__(self,
                 num_convs: int,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: Union[int, List[int], Tuple[int, ...]],
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 initial_stride: Union[int, List[int], Tuple[int, ...]],
                 first: bool = False,
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None
                 ):
        """
        Competitive Dense Block from FastSurfer
        Args:
            num_convs: number of convolution layers (Conv - Dropout - BN - ReLU)
            conv_op: Type of convolution operation (i.e. nn.Conv1d, nn.Conv2d, nn.Conv3d)
            input_channels: number of first layer input-channels
            output_channels:  list of number of filters/features for each convolution output
            kernel_size: Kernel size (same for each convolution in this block)
            initial_stride: First stride of convolution (might be different, if conv is used for pooling)
            first: Input Block to CNN network --> has slightly different processing in beginning (Conv-Norm-Conv)
            conv_bias: If True, adds a learnable bias to the output of the convolution (for nn.Conv1d/2d/3d)
            norm_op: Normalization operation (i.e. BatchNorm - nn.BatchNorm2d/3d, GroupNorm,...)
            norm_op_kwargs: Arguments for Normalization Operation
            dropout_op: Dropout operation (i.e. nn.DropOut)
            dropout_op_kwargs: Arguments for Drop out
            nonlin: Non-linearity
            nonlin_kwargs: Arguments for non-linearity
            nonlin_first: Apply the non-linearity first, then normalization
                          (swap order, normally Conv-Drop-Norm-NonLin --> swap = Conv-Drop-NonLin-Norm)
        """
        super().__init__()
        if not isinstance(output_channels, (tuple, list)):
            output_channels = [output_channels] * num_convs

        stage_modules = []
        start=0
        if first:
            stage_modules.append(InputNormConvNorm(conv_op, input_channels, output_channels[0], kernel_size,
                                                   initial_stride, norm_op, conv_bias, norm_op_kwargs, dropout_op,
                                                   dropout_op_kwargs))
            start=1
            input_channels = output_channels[0]
            logging.debug(f"Input CDB - stage 0: input {input_channels}, output {output_channels[0]}")

        # Add Core RCDNM Block (three in original implementation) -> one conv at end, so num_convs -1 in loop (total=4)
        # Add final RCDN Block = No Maxout in last layer
        for i in range(start, num_convs-1):
            stage_modules.append(ReLUConvDropOutNormMax(conv_op, input_channels, output_channels[i],
                                                        kernel_size, 1, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                                        dropout_op_kwargs, nonlin, nonlin_kwargs, max_last=True
                                                        ))
            logging.debug(f"CDB - stage {i}: input {input_channels}, output {output_channels[i]}")
            input_channels = output_channels[i]
        stage_modules.append(ReLUConvDropOutNormMax(conv_op, output_channels[-2], output_channels[-1], kernel_size,
                                                     1, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                                     dropout_op_kwargs, nonlin, nonlin_kwargs, max_last=False
                                                     )
                             )
        logging.debug(f"CDB - fin stage {num_convs-1}: input {output_channels[-2]}, output {output_channels[-1]}")

        self.convs = nn.Sequential(*stage_modules)
        self.output_channels = output_channels[-1]
        self.initial_stride = maybe_convert_scalar_to_list(conv_op, initial_stride)

    def forward(self, x):
        return self.convs(x)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(
            self.initial_stride), "just give the image size without color/feature channels or " \
                                  "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                  "Give input_size=(x, y(, z))!"
        output = self.convs[0].compute_conv_feature_map_size(input_size)
        size_after_stride = [i // j for i, j in zip(input_size, self.initial_stride)]
        for b in self.convs[1:]:
            output += b.compute_conv_feature_map_size(size_after_stride)
        return output


if __name__ == '__main__':
    data = torch.rand((1, 4, 128, 128, 128))

    encoder = CNNEncoder(input_channels=4,
                         n_stages=5,
                         features_per_stage=(32, 32, 32, 32, 32),
                         conv_op=nn.Conv3d,
                         kernel_sizes=3,
                         strides=(1, 2, 2, 2, 2),
                         n_conv_per_stage=(4, 4, 4, 4, 4),
                         #num_classes=4,
                         #nconv_per_stage_decoder=(2, 2, 2, 2, 2),
                         conv_bias=False,
                         norm_op=nn.BatchNorm3d, norm_op_kwargs=None,
                         dropout_op=None, dropout_op_kwargs=None,
                         nonlin=nn.ReLU, nonlin_kwargs=None,
                         return_skips=True, return_pool_idx=True, pool="max")

    test, idx = encoder.forward(data)
    # 5 skip connections/network outputs), 4 pool idx)
    logging.debug(len(test), len(test), len(idx))
    for i in range(len(test)):
        logging.debug(f"Level {i}: Shape tensor {test[i].shape}")
    #logging.debug(encoder.compute_conv_feature_map_size(data.shape[2:]))

    data = torch.rand((1, 320, 16, 16, 16))

    decoder = CNNDecoder(encoder=encoder, num_classes=8,
                         n_conv_per_stage=4, deep_supervision=True, index_unpool=True)
    fin = decoder.forward(test, idx)
    for layer in fin:
        logging.debug(layer.shape)
    logging.debug(decoder.compute_conv_feature_map_size(fin[0].shape[2:]))