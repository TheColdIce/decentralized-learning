import numpy as np
import random

class Avalanche:
    def __init__(self, connected_nodes, model_params, alpha=0.5, k=10):
        self.alpha = alpha
        self.k = k
        self.connected_nodes = connected_nodes
        self.model_params = model_params
    
    def run_avalanche(self, u):
        success = False
        clients_to_test = self.random_sample(self.connected_nodes, self.k)
        P = self.query(u, clients_to_test)
        if P >= self.alpha * self.k:
            success = True
        return success

    def random_sample(nodes, sample_size):
        return np.random.choice(nodes, min(sample_size, len(nodes)), replace=False)
    
    def query(self, u, clients, tx):
        u_metric = self.test_model(u)
        c_metrics = self.test_model(clients)
        P = 0
        for c_metric in c_metrics:
            if u_metric <= c_metric:
                P = P + 1
        return P
    
    def test_model(self, clients_to_test):
        metrics = []

        for client in clients_to_test:
            client.set_params(self.model_params)
            metrics.append(client.test('test'))
        
        return metrics