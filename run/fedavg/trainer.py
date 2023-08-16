"""federated average trainer"""

from utils import registry
from trainers.FedBaseTrainer import BaseTrainer
from run.fedavg.client import FedAvgClientTrainer, FedAvgClientManager
from run.fedavg.server import FedAvgSyncServerHandler, FedAvgServerManager


@registry.register_fl_algorithm("fedavg")
class FedAvgTrainer(BaseTrainer):
    def __init__(self):
        super().__init__()

        self._before_training()

    def _build_server(self):
        """
        self.handler = FedAvgSyncServerHandler(
            self.model, valid_data=self.data.valid_dataloader,
            test_data=self.data.test_dataloader
        )
        """
        self.handler = FedAvgSyncServerHandler(
            self.server_model, valid_data=self.data.valid_dataloader,
            test_data=self.data.test_dataloader
        )

        self.server_manger = FedAvgServerManager(
            network=self.network,
            handler=self.handler,
        )

    def _build_client(self):
        """
        self.client_trainer = FedAvgClientTrainer(
            model=self.model,
            train_dataset=self.data.train_dataloader_dict,
            valid_dataset=self.data.valid_dataloader_dict,
            # data_slices=self.federated_config.clients_id_list,
        )
        """

        self.client_trainer = FedAvgClientTrainer(
            client_model_rank2=self.client_model_rank2,
            client_model_rank4=self.client_model_rank4,
            client_model_rank8=self.client_model_rank8,
            client_model_rank16=self.client_model_rank16,
            train_dataset=self.data.train_dataloader_dict,
            valid_dataset=self.data.valid_dataloader_dict,
            train_total_step=self.data.train_total_step
            # data_slices=self.federated_config.clients_id_list,
            
        )

        self.client_manager = FedAvgClientManager(
            trainer=self.client_trainer,
            network=self.network
        )
