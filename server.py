import torch
import copy
import random
from typing import List

class Server:
    def __init__(self, model: torch.nn.Module, client_list: List):
        """
        Initialize the Server.
        
        Args:
            model (torch.nn.Module): The global model.
            client_list (List): List of client objects.
        """
        self.global_model = model
        self.clients = client_list
        self.received_models = []
        self.client_losses = {}  

    def select_clients(self, num_clients: int, min_battery: float = 50.0, min_connection: int = 0, min_compute: int = 0) -> List:
        """
        Selects a subset of clients based on battery power, network type, and compute capacity.

        Args:
            num_clients (int): Number of clients to select.
            min_battery (float): Minimum battery power required.
            allowed_networks (List[str]): Allowed network types. 0,1,2
            min_compute (str): Minimum compute capability ("low", "medium", "high"), 0,1,2

        Returns:
            List: List of selected client objects.
        """
        eligible_clients = []
        for client in self.clients:
            info = client.get_client_info()
            conn_map = {"poor": 0, "fair": 1, "good": 2, "excellent": 3}
            comp_map = {"low": 0, "medium": 1, "high": 2}

            conn_value = conn_map.get(info['connection_quality'], 0)
            comp_value = comp_map.get(info['computation_power'], 0)

            if (info['battery'] >= min_battery 
                and conn_value >= min_connection
                and comp_value >= min_compute):
                eligible_clients.append(client)

        selected_clients = random.sample(eligible_clients, min(len(eligible_clients), num_clients))
        return selected_clients

    @staticmethod
    def compute_capacity_sufficient(client_capacity: str, required_capacity: str) -> bool:
        """
        Helper function to compare compute capacities.

        Args:
            client_capacity (str): Capacity of client.
            required_capacity (str): Required minimum capacity.

        Returns:
            bool: True if client meets or exceeds requirement.
        """
        levels = {"low": 0, "medium": 1, "high": 2}
        return levels.get(client_capacity, 0) >= levels.get(required_capacity, 0)

    def send_model(self, client) -> None:
        """
        Sends the global model to a selected client.
        """
        model_copy = copy.deepcopy(self.global_model)
        client.receive_model(model_copy)

    def receive_model(self, updated_model: torch.nn.Module, client_id: int, loss: float) -> None:
        """Receives an updated model from a client."""
        self.received_models.append(updated_model)
        self.client_losses[client_id] = loss

    def aggregate_models(self) -> None:
        """
        Aggregates models received from clients by averaging their parameters.
        """
        if not self.received_models:
            print("Warning: No models received. Skipping aggregation.")
            return
        
        aggregated_model = copy.deepcopy(self.received_models[0])

        with torch.no_grad():
            for param in aggregated_model.parameters():
                param.data.zero_()

            for model in self.received_models:
                for agg_param, client_param in zip(aggregated_model.parameters(), model.parameters()):
                    agg_param.data += client_param.data

            for param in aggregated_model.parameters():
                param.data /= len(self.received_models)

        self.global_model.load_state_dict(aggregated_model.state_dict())
        self.received_models = []  

    def federated_round(self, num_selected_clients: int, min_battery: float = 0.5, min_connection: int = 0, min_compute: int = 0, epochs: int = 1, lr: float = 0.01) -> None:
        """
        Executes one round of Federated Learning.
        """
        selected_clients = self.select_clients(
            num_clients=num_selected_clients,
            min_battery=min_battery,
            min_connection=min_connection,
            min_compute=min_compute
        )

        for idx, client in enumerate(selected_clients):
            self.send_model(client)
            updated_model, train_info = client.train_local_model(epochs=epochs, lr=lr)
            print(f"[Client {idx}] Loss: {train_info['loss']:.4f}, Samples: {train_info['samples_used']}")
            self.receive_model(updated_model, idx, train_info['loss'])

        self.aggregate_models()

    def evaluate_global_model(self, test_loader) -> float:
        """
        Evaluate the global model on a given test dataset.

        Args:
            test_loader (DataLoader): PyTorch DataLoader for test data.

        Returns:
            float: Accuracy of the global model.
        """
        self.global_model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                outputs = self.global_model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100.0 * correct / total
        print(f"Global model accuracy: {accuracy:.2f}%")
        return accuracy


