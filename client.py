
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple
import random



class FederatedClient:
    def __init__(self, data: torch.utils.data.Dataset, model: nn.Module, device: torch.device = torch.device("cpu"), **kwargs):
        """
        Initialize a federated learning client.
        
        Args:
            data: Local training dataset (torch Dataset)
            model: The model architecture to be used (untrained copy)
            device: torch.device
            **kwargs: Client configuration parameters
        """
        self.data = data
        self.device = device
        self.model = model.to(self.device)

      
        self.connection_quality = kwargs.get('connection_quality', 
                                           random.choice(['poor', 'fair', 'good', 'excellent']))
        self.battery = kwargs.get('battery', 
                                  torch.rand(1).item() * 0.8 + 0.2)  
        self.computation_power = kwargs.get('computation_power', 
                                          random.choice(['low', 'medium', 'high']))

    def get_client_info(self) -> Dict:
        """client data for server selection"""
        return {
            'dataset_size': len(self.data),
            'connection_quality': self.connection_quality,
            'battery': self.battery,
            'computation_power': self.computation_power
        }
    
    def receive_model(self, global_model: nn.Module) -> None:
        """
        Receive and load the global model from server.
        """
        self.model.load_state_dict(global_model.state_dict())

    def train_local_model(self, epochs: int = 1, lr: float = 0.01) -> Tuple[nn.Module, Dict]:
        """
        Train the local model on client's dataset.

        Args:
            epochs: Number of training epochs.
            lr: Learning rate.

        Returns:
            Tuple of (trained model, training metadata)
        """
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        train_loader = torch.utils.data.DataLoader(self.data, batch_size=8, shuffle=True)

        total_loss = 0.0
        total_samples = 0

        for epoch in range(epochs):
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

        return self.model, {
            'loss': avg_loss,
            'samples_used': total_samples
        }





