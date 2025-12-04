import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.activations import PhotonicSigmoid
from utils.activations2 import EluLikeAF, ReSinAF
class LinearDenoiser(nn.Module):
    def __init__(self, input_size=33, output_size=32, num_classes=None):
        """
        Multi-task autoencoder with optional classification head.

        Args:
            input_size: Input dimension
            output_size: Output dimension (reconstruction)
            num_classes: Number of classes for classification. If None, no classifier head.
        """
        super(LinearDenoiser, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.num_classes = num_classes

        # self.photonic_sigmoid = PhotonicSigmoid()
        self.elu_like_af = EluLikeAF.from_obs(3)
        # self.resin_af = ReSinAF.from_obs(150)

        ## encoder layers ##
        self.linear1 = nn.Linear(input_size, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(0.2)

        self.linear2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.dropout2 = nn.Dropout(0.2)

        # Bottleneck layer (latent representation)
        self.linear3 = nn.Linear(16, 16)
        self.bn3 = nn.BatchNorm1d(16)

        ## decoder layers ##
        self.linear4 = nn.Linear(16, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(0.2)

        self.linear5 = nn.Linear(32, 32) # TODO 128 -> 32, 64 -> 32 --> if it works -> try to remove linear layers
        self.bn5 = nn.BatchNorm1d(32)
        self.dropout4 = nn.Dropout(0.2)

        self.linear_out = nn.Linear(32, output_size)

        if num_classes is not None:
            self.classifier = Classifier(input_length=8, num_classes=num_classes)
        else:
            self.classifier = None

    def encode(self, x):
        """Encode input to latent representation."""
        if len(x.shape) == 3:
            x = x.squeeze(1)

        # ## encoder ##
        # x = F.gelu(self.bn1(self.linear1(x))) # TODO gelu -> "photonic sigmoid"
        # x = self.dropout1(x)

        # x = F.gelu(self.bn2(self.linear2(x)))
        # x = self.dropout2(x)

        # # Store 64-dim intermediate representation for classification
        # latent_features = x.clone()

        # # Bottleneck (32-dim compressed representation)
        # bottleneck = F.gelu(self.bn3(self.linear3(x)))

        ## encoder ##
        x = self.elu_like_af(self.bn1(self.linear1(x)))
        x = self.dropout1(x)

        x = self.elu_like_af(self.bn2(self.linear2(x)))
        x = self.dropout2(x)

        # Store 64-dim intermediate representation for classification
        latent_features = x.clone()

        bottleneck = self.elu_like_af(self.bn3(self.linear3(x)))

        return bottleneck, latent_features

    def decode(self, bottleneck):
        """Decode bottleneck representation to output."""
        
        # ## decoder ##
        # x = F.gelu(self.bn4(self.linear4(bottleneck)))
        # x = self.dropout3(x)

        # x = F.gelu(self.bn5(self.linear5(x)))
        # x = self.dropout4(x)

        ## decoder ##
        x = self.elu_like_af(self.bn4(self.linear4(bottleneck)))
        x = self.dropout3(x)

        x = self.elu_like_af(self.bn5(self.linear5(x)))
        x = self.dropout4(x)

        # Output layer (no activation for reconstruction)
        x = self.linear_out(x)
        x = x.unsqueeze(1)

        return x

    def forward(self, x):
        """
        Forward pass.

        Returns:
            If classifier is None: (reconstruction, latent_features)
            If classifier is not None: (reconstruction, latent_features, class_logits)

            Note: latent_features is the 64-dim intermediate representation (before bottleneck)
                  used for classification tasks.
        """
        # Encode
        bottleneck, latent_features = self.encode(x)

        # Decode
        reconstruction = self.decode(bottleneck)

        # Classify (if classifier exists) - use 64-dim latent_features for classification
        if self.classifier is not None:
            # Reshape latent_features to [batch, 1, 64] for CNN classifier
            # latent_features_reshaped = latent_features.unsqueeze(1)
            latent_output_reshaped = bottleneck.unsqueeze(1)
            class_logits = self.classifier(latent_output_reshaped)
            # return reconstruction, latent_features, class_logits
            return reconstruction, bottleneck, class_logits
        else:
            # return reconstruction, latent_features
            return reconstruction, bottleneck
    
class Classifier(nn.Module):
    def __init__(self, input_length=33, num_classes=4):
        super(Classifier, self).__init__()
        
        # self.photonic_sigmoid = PhotonicSigmoid()
        self.elu_like_af = EluLikeAF.from_obs(3)

        # Use adaptive pooling for cleaner size handling
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1) #Todo when i will have 10 chips as input the in_channels=10 instead of 1
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.AdaptiveMaxPool1d(input_length // 2)
        # Lout = 16
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.AdaptiveMaxPool1d(input_length // 4)
        # Lout = 8
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.AdaptiveMaxPool1d(input_length // 8)
        # Lout = 4
        
        # Global average pooling to handle any input size
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, z):
        z = self.pool1(F.elu(self.bn1(self.conv1(z))))  # ELU for conv layers
        z = self.pool2(F.elu(self.bn2(self.conv2(z))))
        z = self.pool3(F.elu(self.bn3(self.conv3(z))))

        # z = self.pool1(self.elu_like_af(self.bn1(self.conv1(z))))  # ELU for conv layers
        # z = self.pool2(self.elu_like_af(self.bn2(self.conv2(z))))
        # z = self.pool3(self.elu_like_af(self.bn3(self.conv3(z))))
        
        z = self.global_pool(z)  # Global pooling to size [batch, 128, 1]
        z = z.view(z.size(0), -1)  # Flatten to [batch, 128]
        
        z = F.gelu(self.fc1(z))  # GELU for FC layers

        # z = self.elu_like_af(self.fc1(z))  # GELU for FC layers

        z = self.dropout(z)
        z = self.fc2(z)  # No activation - CrossEntropyLoss expects logits

        return z