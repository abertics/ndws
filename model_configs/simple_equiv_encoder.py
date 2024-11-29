import torch
import torch.nn as nn
import torch.nn.functional as F
import e2cnn.nn as enn
import e2cnn.gspaces as gspaces
from e2cnn.nn.init import generalized_he_init

import numpy as np
import math

class AACN_Layer(nn.Module):
    def __init__(self, in_channels,out_channels, k=0.25, v=0.25, kernel_size=3, num_heads=8, image_size=224, inference=False):
        super(AACN_Layer, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.dk = math.floor((in_channels * k) / num_heads) * num_heads
        # Ensure a minimum of 20 dimensions per head for the keys
        if self.dk / num_heads < 20:
            self.dk = num_heads * 20
        self.dv = math.floor((in_channels * v) / num_heads) * num_heads

        assert self.dk % self.num_heads == 0, "dk should be divisible by num_heads."
        assert self.dv % self.num_heads == 0, "dv should be divisible by num_heads."

        self.padding = (self.kernel_size - 1) // 2

        # Modify conv_out to output 1 channel
        self.conv_out = nn.Conv2d(self.in_channels, out_channels, self.kernel_size, padding=self.padding)
        # Adjust kqv_conv accordingly
        self.kqv_conv = nn.Conv2d(self.in_channels, 2 * self.dk + self.dv, kernel_size=1)
        # Modify attn_out to output 1 channel
        self.attn_out = nn.Conv2d(self.dv, out_channels, 1)

        # Positional encodings
        self.rel_encoding_h = nn.Parameter(
            torch.randn((2 * image_size - 1, self.dk // self.num_heads), requires_grad=True)
        )
        self.rel_encoding_w = nn.Parameter(
            torch.randn((2 * image_size - 1, self.dk // self.num_heads), requires_grad=True)
        )

        # Optionally store attention weights
        self.inference = inference
        if self.inference:
            self.register_parameter('weights', None)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        dkh = self.dk // self.num_heads
        dvh = self.dv // self.num_heads
        flatten_hw = lambda x, depth: torch.reshape(x, (batch_size, self.num_heads, height * width, depth))

        # Compute q, k, v
        kqv = self.kqv_conv(x)
        k, q, v = torch.split(kqv, [self.dk, self.dk, self.dv], dim=1)
        q = q * (dkh ** -0.5)

        # Split heads
        k = self.split_heads_2d(k, self.num_heads)
        q = self.split_heads_2d(q, self.num_heads)
        v = self.split_heads_2d(v, self.num_heads)

        # Compute attention logits
        qk = torch.matmul(flatten_hw(q, dkh), flatten_hw(k, dkh).transpose(2, 3))

        # Add relative logits
        qr_h, qr_w = self.relative_logits(q)
        qk += qr_h
        qk += qr_w

        # Compute attention weights
        weights = F.softmax(qk, dim=-1)
        if self.inference:
            self.weights = nn.Parameter(weights)

        # Compute attention output
        attn_out = torch.matmul(weights, flatten_hw(v, dvh))
        attn_out = torch.reshape(attn_out, (batch_size, self.num_heads, dvh, height, width))
        attn_out = self.combine_heads_2d(attn_out)
        attn_out = self.attn_out(attn_out)

        # Compute conv_out
        conv_out = self.conv_out(x)

        # Sum conv_out and attn_out to produce output of shape (B, 1, H, W)
        return conv_out + attn_out

    # Split channels into multiple heads
    def split_heads_2d(self, inputs, num_heads):
        batch_size, depth, height, width = inputs.size()
        ret_shape = (batch_size, num_heads, height, width, depth // num_heads)
        split_inputs = torch.reshape(inputs, ret_shape)
        return split_inputs

    # Combine heads (inverse of split_heads_2d)
    def combine_heads_2d(self, inputs):
        batch_size, num_heads, depth, height, width = inputs.size()
        ret_shape = (batch_size, num_heads * depth, height, width)
        return torch.reshape(inputs, ret_shape)

    # Compute relative logits for both dimensions
    def relative_logits(self, q):
        _, num_heads, height, width, dkh = q.size()
        rel_logits_w = self.relative_logits_1d(
            q, self.rel_encoding_w, height, width, num_heads, [0, 1, 2, 4, 3, 5]
        )
        rel_logits_h = self.relative_logits_1d(
            torch.transpose(q, 2, 3), self.rel_encoding_h, width, height, num_heads, [0, 1, 4, 2, 5, 3]
        )
        return rel_logits_h, rel_logits_w

    # Compute relative logits along one dimension
    def relative_logits_1d(self, q, rel_k, height, width, num_heads, transpose_mask):
        rel_logits = torch.einsum('bhxyd,md->bxym', q, rel_k)
        # Collapse height and heads
        rel_logits = torch.reshape(rel_logits, (-1, height, width, 2 * width - 1))
        rel_logits = self.rel_to_abs(rel_logits)
        # Shape it
        rel_logits = torch.reshape(rel_logits, (-1, height, width, width))
        # Tile for each head
        rel_logits = torch.unsqueeze(rel_logits, dim=1)
        rel_logits = rel_logits.repeat(1, num_heads, 1, 1, 1)
        # Tile height / width times
        rel_logits = torch.unsqueeze(rel_logits, dim=3)
        rel_logits = rel_logits.repeat(1, 1, 1, height, 1, 1)
        # Reshape for adding to the logits
        rel_logits = rel_logits.permute(transpose_mask)
        rel_logits = torch.reshape(rel_logits, (-1, num_heads, height * width, height * width))
        return rel_logits

    # Converts tensor from relative to absolute indexing
    def rel_to_abs(self, x):
        batch_size, num_heads, L, _ = x.size()
        # Pad to shift from relative to absolute indexing
        col_pad = torch.zeros((batch_size, num_heads, L, 1), device=x.device)
        x = torch.cat((x, col_pad), dim=3)
        flat_x = torch.reshape(x, (batch_size, num_heads, L * 2 * L))
        flat_pad = torch.zeros((batch_size, num_heads, L - 1), device=x.device)
        flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)
        # Reshape and slice out the padded elements
        final_x = torch.reshape(flat_x_padded, (batch_size, num_heads, L + 1, 2 * L - 1))
        final_x = final_x[:, :, :L, L - 1 :]
        return final_x
    

class EquivariantResidualBlock(nn.Module):
    def __init__(self, in_type, out_type, dropout_prob):
        super(EquivariantResidualBlock, self).__init__()
        self.conv1 = enn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False)
        self.bn1 = enn.InnerBatchNorm(out_type)
        self.relu = enn.ReLU(out_type, inplace=True)
        self.dropout = enn.PointwiseDropout(out_type, p=dropout_prob)
        self.conv2 = enn.R2Conv(out_type, out_type, kernel_size=3, padding=1, bias=False)
        self.bn2 = enn.InnerBatchNorm(out_type)

        # If in_type and out_type are different, use a projection
        if in_type != out_type:
            self.downsample = enn.R2Conv(in_type, out_type, kernel_size=1, bias=False)
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity  # Element-wise addition of GeometricTensors
        out = self.relu(out)
        return out
    
class EquivariantVectorEncoder(nn.Module):
    def __init__(self, in_channels, conv_channels, dropout_prob):
        super(EquivariantVectorEncoder, self).__init__()
        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.dropout_prob = dropout_prob

        # Define the rotational symmetry group
        self.r2_act = gspaces.Rot2dOnR2(N=8)  # N=8 for 8-fold rotational symmetry

        # Encoder
        self.encoder_layers = nn.ModuleList()
        prev_type = enn.FieldType(self.r2_act, in_channels * [self.r2_act.trivial_repr])
        self.encoder_types = []
        for out_channels in conv_channels:
            out_type = enn.FieldType(self.r2_act, out_channels * [self.r2_act.regular_repr])
            self.encoder_types.append(out_type)
            self.encoder_layers.append(
                nn.Sequential(
                    EquivariantResidualBlock(prev_type, out_type, dropout_prob),
                    enn.GroupPooling(out_type)
                )
            )
            prev_type = enn.FieldType(self.r2_act, out_channels * [self.r2_act.trivial_repr])
            self.encoder_layers.append(enn.PointwiseMaxPool(prev_type, kernel_size=2, stride=2))

        self.embed_dim = prev_type.size  # Number of channels after encoding
        
        # Add global average pooling to get a vector representation
        self.global_pool = enn.GroupPooling(prev_type)

    def forward(self, x):
        # Ensure x is a torch.Tensor with the correct dtype and device
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
        elif x.dtype != torch.float32:
            x = x.float()
        
        # Wrap input in GeometricTensor
        input_type = enn.FieldType(self.r2_act, self.in_channels * [self.r2_act.trivial_repr])
        x = enn.GeometricTensor(x, input_type)

        # Encoder
        for layer in self.encoder_layers:
            x = layer(x)
            
        # Final pooling to reduce spatial dimensions
        x = F.avg_pool2d(x.tensor, kernel_size=2, stride=2)
        
        # # Flatten to vector
        # B, C, H, W = x.shape
        # x = x.view(B, -1)  # Flatten spatial and channel dimensions

        return x  # Return flattened vector [B, C*H*W]

class EquivariantVectorDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, conv_channels, dropout_prob):
        super(EquivariantVectorDecoder, self).__init__()
        self.in_channels = in_channels  # Input channels from latent representation
        self.out_channels = out_channels  # Number of channels in output image
        self.conv_channels = conv_channels
        self.dropout_prob = dropout_prob

        # Define the rotational symmetry group
        self.r2_act = gspaces.Rot2dOnR2(N=8)  # N=8 for 8-fold rotational symmetry

        # Decoder
        self.decoder_layers = nn.ModuleList()
        prev_type = enn.FieldType(self.r2_act, self.in_channels * [self.r2_act.trivial_repr])
        self.decoder_types = []
        for out_channels in reversed(conv_channels):
            out_type = enn.FieldType(self.r2_act, out_channels * [self.r2_act.regular_repr])
            self.decoder_types.append(out_type)
            self.decoder_layers.append(
                nn.Sequential(
                    # Upsample
                    enn.R2Upsampling(prev_type, scale_factor=2, mode='bilinear'),
                    # Equivariant Residual Block
                    EquivariantResidualBlock(prev_type, out_type, self.dropout_prob)
                )
            )
            prev_type = out_type  # Update prev_type for the next layer

        # Final layer to map to desired output channels
        final_type = enn.FieldType(self.r2_act, self.out_channels * [self.r2_act.trivial_repr])
        self.final_layer = enn.R2Conv(prev_type, final_type, kernel_size=3, padding=1)

    def forward(self, x):
        # Ensure x is a torch.Tensor with the correct dtype and device
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=x.device)
        elif x.dtype != torch.float32:
            x = x.float()

        # Wrap input in GeometricTensor
        input_type = enn.FieldType(self.r2_act, self.in_channels * [self.r2_act.trivial_repr])
        x = enn.GeometricTensor(x, input_type)

        # Decoder
        for layer in self.decoder_layers:
            x = layer(x)

        x = self.final_layer(x)

        return x.tensor  # Return output image tensor


class EquivAutoencoder(nn.Module):
    def __init__(self, in_channels,conv_channels, dropout_prob,out_channels = 7):
        super(EquivAutoencoder, self).__init__()

        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.dropout_prob = dropout_prob

        self.out_channels = out_channels
        

        self.encoder = EquivariantVectorEncoder(self.in_channels,conv_channels,dropout_prob)

        self.encoded_channels = self.conv_channels[-1]
        self.decoder_channels = [self.encoded_channels//2]+self.conv_channels
        self.decoder = EquivariantVectorDecoder(self.encoded_channels,self.out_channels,self.decoder_channels,self.dropout_prob)

        self.final2 = AACN_Layer(self.out_channels+1,1, k=0.25, v=0.25, kernel_size=3, num_heads=2, image_size=32, inference=False)
        self.final1 = AACN_Layer(self.out_channels+1,self.out_channels+1, k=0.25, v=0.25, kernel_size=3, num_heads=2, image_size=32, inference=False)

    def forward(self,x):
        prev_fire_mask = x[:,-1,...].unsqueeze(1)
        
        encoded = self.encoder(x) # [B,C,H,W]

        # print('combined shape',combined.shape)
        decoded = self.decoder(encoded) # [B,C,H,W]

        decoded = torch.cat((decoded,prev_fire_mask),dim = 1) # [B,C+1,H,W]

        decoded = self.final1(decoded) # [B,C+1,H,W]

        preds = self.final2(decoded) # [B,1,H,W]

        # # Split into difference prediction and gates
        # diff_pred, gates = torch.chunk(diff_and_gates, 2, dim=1)
        # alpha = torch.sigmoid(gates)
        # beta = torch.sigmoid(diff_pred)
        
        # new_mask = prev_fire_mask * (1 - alpha) + beta
        return preds  # No clamp needed since both terms are now bounded [0,1]

def init_weights(m):
    if isinstance(m, enn.R2Conv):
        # Use generalized He initialization for equivariant layers
        generalized_he_init(m.weights.data, m.basisexpansion, cache=True)
        
        # Initialize bias to zero if it exists
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        # Standard He initialization for regular Conv2d layers
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


# def split_features(features):
#     fuel_features = ['impervious','water','fuel1','fuel2','fuel3',
#                  'NDVI','pdsi','pr','erc','bi','avg_sph',
#                  'tmp_day','tmp_75','viirs_PrevFireMask']
#     dynamics_features = ['elevation','chili','population','gust_med',
#                      'wind_avg','wind_75','wdir_wind','wdir_gust',
#                      'viirs_PrevFireMask']
#     fuel_inds = []
#     dynamics_inds = []
#     for i,x in enumerate(features):
#         if x in fuel_features:
#             fuel_inds.append(i)
#         if x in dynamics_features:
#             dynamics_inds.append(i)

#     fuel_inds = np.array(fuel_inds)
#     dynamics_inds = np.array(dynamics_inds)
    
#     return fuel_inds,dynamics_inds

def load(config):

	torch.manual_seed(42)

	model = EquivAutoencoder(
		in_channels = config['in_channels'],
		conv_channels = config['conv_channels'],
		dropout_prob = config['dropout_prob'],
        # features = config['features'],
        out_channels = config['final_channels']
	)

	model.apply(init_weights)

	return model


