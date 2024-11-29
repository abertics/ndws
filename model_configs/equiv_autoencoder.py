import torch
import torch.nn as nn
import torch.nn.functional as F
import e2cnn.nn as enn
import e2cnn.gspaces as gspaces
from e2cnn.nn.init import generalized_he_init

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
        # out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        # out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity  # Element-wise addition of GeometricTensors
        out = self.relu(out)
        return out

class GeometricTensorWrapper(nn.Module):
    def __init__(self, field_type):
        super().__init__()
        self.field_type = field_type
    
    def forward(self, x):
        # If x is already a GeometricTensor, get its underlying tensor
        if hasattr(x, 'tensor'):
            x = x.tensor
        # Now x should be a torch.Tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        return enn.GeometricTensor(x, self.field_type)

class ConvAutoencoder(nn.Module):
    def __init__(self, in_channels, conv_channels, num_attention_blocks, dropout_prob):
        super(ConvAutoencoder, self).__init__()
        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.num_attention_blocks = num_attention_blocks
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

        # Bottleneck Attention Blocks
        self.embed_dim = prev_type.size  # Number of channels after encoding
        self.attention_blocks = nn.ModuleList()
        for _ in range(num_attention_blocks):
            self.attention_blocks.append(
                nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=8, dropout=dropout_prob)
            )

        # Decoder (simplified without skip connections)
        self.decoder_layers = nn.ModuleList()
        conv_channels_rev = conv_channels[::-1]
        prev_type = enn.FieldType(self.r2_act, (prev_type.size // self.r2_act.regular_repr.size) * [self.r2_act.regular_repr])
        
        for out_channels in conv_channels_rev:
            out_type = enn.FieldType(self.r2_act, out_channels * [self.r2_act.regular_repr])
            
            self.decoder_layers.append(
                nn.Sequential(
                    enn.R2Upsampling(prev_type, scale_factor=2, mode='bilinear', align_corners=False),
                    enn.R2Conv(prev_type, out_type, kernel_size=3, padding=1, bias=False),
                    enn.InnerBatchNorm(out_type),
                    enn.ReLU(out_type, inplace=True),
                    enn.PointwiseDropout(out_type, p=dropout_prob)
                )
            )
            prev_type = out_type

        # Final Convolution Layer to output scalar field
        self.final_conv = nn.Conv2d(prev_type.size, 1, kernel_size=1, bias=True)

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
        encoder_outputs = []
        for i in range(0, len(self.encoder_layers), 2):
            x = self.encoder_layers[i](x)  # EquivariantResidualBlock + GroupPooling
            encoder_outputs.append(x)
            x = self.encoder_layers[i+1](x)  # MaxPool

        # Flatten for Attention Blocks
        B, C, H, W = x.tensor.shape
        x_flat = x.tensor.view(B, C, -1).permute(2, 0, 1)  # Shape: (S, B, C), where S = H*W

        # Bottleneck with Multi-Head Self-Attention
        for attn in self.attention_blocks:
            x_flat, _ = attn(x_flat, x_flat, x_flat)

        # Reshape back to spatial dimensions
        x_tensor = x_flat.permute(1, 2, 0).view(B, C, H, W)
        
        # Wrap tensor back as GeometricTensor before decoder
        num_fields = C // self.r2_act.regular_repr.size
        decoder_input_type = enn.FieldType(self.r2_act, num_fields * [self.r2_act.regular_repr])
        x = enn.GeometricTensor(x_tensor, decoder_input_type)

        # Decoder (simplified without skip connections)
        for layer in self.decoder_layers:
            x = layer(x)

        # Final Output Layer
        x = self.final_conv(x.tensor)
        return x

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


def load(config):

	torch.manual_seed(42)

	model = ConvAutoencoder(
		in_channels = config['in_channels'],
		conv_channels = config['conv_channels'],
		num_attention_blocks = config['num_attention_blocks'],
		dropout_prob = config['dropout_prob']
	)

	model.apply(init_weights)

	return model


