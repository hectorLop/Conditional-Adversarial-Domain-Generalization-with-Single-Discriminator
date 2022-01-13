import torch
import torch.nn as nn
import timm

class AdversarialNetwork(nn.Module):
    """
    Network implementing an adversarial arquitecture.

    Args:
        discriminator (nn.Module): Discriminator network.
        classifier (nn.Module): Classifier network.
        extractor_name (str): CNN feature extractor model name.
        feat_size (int): Feature extractor output size.
        in_chans (int): Feature extractor input channel.
    """

    def __init__(
        self,
        discriminator : nn.Module,
        classifier : nn.Module,
        extractor_name : str = 'resnet50',
        feat_size : int = 2048,
        in_chans : int = 1
    ) -> None:
        super().__init__()
        
        # Create a cnn model to be used as feature extractor
        cnn_model = timm.create_model(extractor_name,
                                      pretrained=True,
                                      in_chans=in_chans,
                                      num_classes=0)
        # Feature extractor module                                      
        self.feature_extractor = nn.Sequential(
            cnn_model,
            nn.Linear(feat_size, 256),
            nn.ReLU()
        )
        
        self.discriminator = discriminator
        self.classifier = classifier
        
    def forward_disc(self, x : torch.Tensor) -> torch.Tensor:
        """
        Discriminator forward pass
        """
        x = self.feature_extractor(x)
        
        disc_out = self.discriminator(x)
        
        return disc_out
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Classifier forward pass
        """
        x = self.feature_extractor(x)
        
        class_out = self.classifier(x)
        
        return class_out
    
class Classifier(nn.Module):
    """
    Network implementing the Classifier module.

    Args:
        feature_size (int): Classifier output size.
        target_size (int): Feature extractor input channel.
    """

    def __init__(self, feature_size : int, target_size : int):
        super().__init__()
                
        self.model = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, target_size),
        )
        
    def forward(self, x):
        x = self.model(x)
        
        return x
    
class Discriminator(nn.Module):
    """
    Network implementing the Classifier module.

    Args:
        feature_size (int): Classifier output size.
        dropout_ps (float): Dropout ratio.
        n_domains (int): Number of domains.
        n_clases (int): Number of classes
    """
    
    def __init__(
        self,
        feature_size : int,
        dropout_ps : float,
        n_domains : int,
        n_classes : int
    ) -> None:
        super().__init__()
        
        output_size = n_domains * n_classes
        
        self.disc = nn.Sequential(
            nn.Linear(feature_size, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_ps),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_ps),
            nn.Linear(1024, output_size),
        )
        
    def forward(self, x):
        x = self.disc(x)
        
        return x