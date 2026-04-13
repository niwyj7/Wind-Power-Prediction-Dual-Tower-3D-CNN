import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        # Note: Ensure mid_channels is at least 1 to prevent errors if channels < reduction_ratio
        mid_channels = max(1, channels // reduction_ratio) 
        self.fc = nn.Sequential(
            nn.Linear(channels, mid_channels, bias=False),
            nn.ReLU(),
            nn.Linear(mid_channels, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        
        # Multiply the attention weights with the original feature map
        return x * self.sigmoid(out).view(b, c, 1, 1, 1)

class DualTowerGridCNN3D(nn.Module):
    def __init__(self, input_shape):
        super(DualTowerGridCNN3D, self).__init__()
        
        # Extract input channels
        in_channels = input_shape[0]
        
        # South feature extraction tower
        self.tower_south = nn.Sequential(
            nn.Conv3d(in_channels, 8, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)), 
            nn.Conv3d(8, 16, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU()
        )
        
        # North feature extraction tower
        self.tower_north = nn.Sequential(
            nn.Conv3d(in_channels, 8, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(8, 16, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU()
        )

        # Channel Attention modules
        self.ca_south = ChannelAttention(16)
        self.ca_north = ChannelAttention(16)

        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Fully connected layers (South)
        self.fc_south = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Softplus()
        )

        # Fully connected layers (North)
        self.fc_north = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Softplus()  
        )
    def forward(self, x_south, x_north):
        # Process South stream
        s = self.tower_south(x_south)
        s = self.ca_south(s)
        s = self.gap(s)
        out_south = self.fc_south(s)

        # Process North stream
        n = self.tower_north(x_north)
        n = self.ca_north(n)
        n = self.gap(n)
        out_north = self.fc_north(n)
        
        # Fused output (element-wise addition)
        total_output = out_south + out_north
        return total_output
