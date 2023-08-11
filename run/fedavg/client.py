"""federated average client"""

from abc import ABC
from trainers.BaseClient import BaseClientTrainer, BaseClientManager


class FedAvgClientTrainer(BaseClientTrainer, ABC):
    """
    def __init__(self, model, train_dataset, valid_dataset):
        super().__init__(model, train_dataset, valid_dataset)
    """
    def __init__(self, client_model_rank2, client_model_rank4, client_model_rank8, client_model_rank16, train_dataset, valid_dataset):
        super().__init__(client_model_rank2, client_model_rank4, client_model_rank8, client_model_rank16, train_dataset, valid_dataset)



class FedAvgClientManager(BaseClientManager, ABC):
    def __init__(self, network, trainer):
        super().__init__(network, trainer)
