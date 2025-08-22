import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class OpenMaxClassifier(nn.Module):
    def __init__(self, input_dim: int, num_labels: int, hidden: int = 128, p: float = 0.1, alpha: float = 20.0):
        super().__init__()
        self.num_labels = num_labels
        self.alpha = alpha
        
        # Main classifier network
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(256, hidden),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden, num_labels),
        )
        
        # Weibull parameters for each class (learned during training)
        self.register_buffer('weibull_alpha', torch.ones(num_labels))
        self.register_buffer('weibull_beta', torch.ones(num_labels))
        self.register_buffer('weibull_tau', torch.zeros(num_labels))
        
        # Activation vectors for each class (learned during training)
        self.register_buffer('activation_vectors', torch.zeros(num_labels, hidden))
        
    def forward(self, x: torch.Tensor, use_openmax: bool = True) -> torch.Tensor:
        # Get activations from the penultimate layer
        activations = x
        for i, layer in enumerate(self.net[:-1]):
            activations = layer(activations)
        
        # Get logits from final layer
        logits = self.net[-1](activations)
        
        if use_openmax and self.training == False:
            # Apply OpenMax during inference
            return self.openmax_forward(activations, logits)
        else:
            return logits
    
    def openmax_forward(self, activations: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply OpenMax to get calibrated probabilities with unknown class
        """
        batch_size = activations.size(0)
        
        # Calculate distances to activation vectors for each class
        distances = torch.zeros(batch_size, self.num_labels, device=activations.device)
        for i in range(self.num_labels):
            class_activation = self.activation_vectors[i]
            distances[:, i] = torch.norm(activations - class_activation, dim=1)
        
        # Apply Weibull CDF to get probability of being unknown
        unknown_probs = torch.zeros(batch_size, device=activations.device)
        for i in range(self.num_labels):
            # Weibull CDF: 1 - exp(-((x-tau)/beta)^alpha)
            x = distances[:, i]
            alpha = self.weibull_alpha[i]
            beta = self.weibull_beta[i]
            tau = self.weibull_tau[i]
            
            # Avoid division by zero and negative values
            safe_beta = torch.clamp(beta, min=1e-6)
            safe_x = torch.clamp(x - tau, min=0)
            
            weibull_cdf = 1 - torch.exp(-torch.pow(safe_x / safe_beta, alpha))
            unknown_probs = torch.maximum(unknown_probs, weibull_cdf)
        
        # Adjust logits based on unknown probability
        adjusted_logits = logits.clone()
        for i in range(batch_size):
            unknown_prob = unknown_probs[i]
            if unknown_prob > 0.5:  # High uncertainty threshold
                # Reduce confidence in all classes
                adjusted_logits[i] = adjusted_logits[i] * (1 - unknown_prob)
        
        return adjusted_logits
    
    def fit_weibull(self, activations: torch.Tensor, labels: torch.Tensor):
        """
        Fit Weibull distribution parameters for each class
        This should be called after training with validation data
        """
        print("Fitting Weibull distributions for OpenMax...")
        
        for class_idx in range(self.num_labels):
            # Get activations for this class
            class_mask = labels == class_idx
            if class_mask.sum() == 0:
                continue
                
            class_activations = activations[class_mask]
            
            # Calculate mean activation vector for this class
            mean_activation = class_activations.mean(dim=0)
            self.activation_vectors[class_idx] = mean_activation
            
            # Calculate distances to mean activation
            distances = torch.norm(class_activations - mean_activation, dim=1)
            
            # Fit Weibull parameters (simplified fitting)
            # In practice, you'd use a proper Weibull fitting library
            distances_np = distances.cpu().numpy()
            
            # Simple parameter estimation
            self.weibull_alpha[class_idx] = torch.tensor(2.0)  # Shape parameter
            self.weibull_beta[class_idx] = torch.tensor(distances_np.std())  # Scale parameter
            self.weibull_tau[class_idx] = torch.tensor(distances_np.min())  # Location parameter
            
        print("Weibull fitting completed!")


# Keep the old classifier for backward compatibility
class Classifier(nn.Module):
    def __init__(self, input_dim: int, num_labels: int, hidden: int = 128, p: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(256, hidden),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden, num_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
