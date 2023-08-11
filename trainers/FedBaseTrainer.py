"""BaseTrainer for FedETuning"""

from abc import ABC
from utils import registry
from utils import setup_seed
from utils import global_metric_save
from fedlab.core.network import DistNetwork


class BaseTrainer(ABC):
    def __init__(self, *args):

        config = registry.get("config")
        self.model_config = config.M
        self.data_config = config.D
        self.training_config = config.T
        self.federated_config = config.F

        self.logger = registry.get("logger")

        # self._before_training()

    @property
    def role(self):
        if self.federated_config.rank == 0:
            return "server"
        elif self.federated_config.rank > 0:
            return f"sub-server_{self.federated_config.rank}"
        else:
            return "centralized"

    def _build_server(self):
        raise NotImplementedError

    def _build_client(self):
        raise NotImplementedError

    def _build_local_trainer(self, *args):
        raise NotImplementedError

    def _build_network(self):
        self.network = DistNetwork(
            address=(self.federated_config.ip, self.federated_config.port),
            world_size=self.federated_config.world_size,
            rank=self.federated_config.rank,
            ethernet=self.federated_config.ethernet)

    def _build_data(self):
        self.data = registry.get_data_class(self.data_config.dataset_name)()

    """
    def _build_model(self):
        self.model = registry.get_model_class(self.model_config.model_output_mode)(
            task_name=self.data_config.task_name
        )
        # print("inside _build_model self.model")
        # print(self.model)
        # for key, backbone_layer in backbone.named_parameters():
            # print("index:", index, " key:", key)
        # print("martinc quit---------------------------------------")
        # quit()
    """
    def _build_server_model(self):

        #martinc todo
        delta_args = registry.get("delta_config")
        lora_r = delta_args["lora_r"]
        lora_alpha = delta_args["lora_alpha"]

        self.server_model = registry.get_model_class(self.model_config.model_output_mode)(# cls, name, lora_r, lora_alpha
            task_name=self.data_config.task_name,
            lora_r=lora_r,
            lora_alpha=lora_alpha
        )

    def _build_client_model(self, server_rank):
        #martinc todo
        lora_r = 2
        lora_alpha = 2

        self.client_model_rank2 = registry.get_model_class(self.model_config.model_output_mode)(# cls, name, lora_r, lora_alpha
            task_name=self.data_config.task_name,
            lora_r=lora_r,
            lora_alpha=lora_alpha
        )

        lora_r = 4
        lora_alpha = 4

        self.client_model_rank4 = registry.get_model_class(self.model_config.model_output_mode)(# cls, name, lora_r, lora_alpha
            task_name=self.data_config.task_name,
            lora_r=lora_r,
            lora_alpha=lora_alpha
        )

        lora_r = 8
        lora_alpha = 8

        self.client_model_rank8 = registry.get_model_class(self.model_config.model_output_mode)(# cls, name, lora_r, lora_alpha
            task_name=self.data_config.task_name,
            lora_r=lora_r,
            lora_alpha=lora_alpha
        )

        # if (server_rank >= 16):
        lora_r = 16
        lora_alpha = 16
        self.client_model_rank16 = registry.get_model_class(self.model_config.model_output_mode)(# cls, name, lora_r, lora_alpha
            task_name=self.data_config.task_name,
            lora_r=lora_r,
            lora_alpha=lora_alpha
        )

    def _before_training(self):

        self.logger.info(f"{self.role} set seed {self.training_config.seed}")
        setup_seed(self.training_config.seed)

        self.logger.info(f"{self.role} building dataset ...")
        # set before build model
        self._build_data()

        # self.logger.info(f"{self.role} building local trainer ...")
        # self._build_local_trainer()

        if self.federated_config.rank != -1:
            self.logger.info(f"{self.role} building network ...")
            self._build_network()

        if self.federated_config.rank == 0:
            self.logger.info("martinc building server ...")
            self.logger.info(f"{self.role} building model ...")

            # need to build model server rank8
            # self._build_model()
            # martinc
            self._build_server_model()

            #print("self.server_model_rank8")
            #print(self.server_model_rank8)
            #print("martinc fff quit------------------")
            #quit()
            self.logger.info("building server ...")
            self._build_server()
        else:
            self.logger.info("martinc building client ...")
            self.logger.info(f"{self.role} building model client rank2, rank4, rank8 ...")
            
            delta_args = registry.get("delta_config")
            self.server_rank = delta_args["lora_r"]
            # need to build model client rank2, rank4, rank8
            # self._build_model()
            # martinc
            self._build_client_model(self.server_rank)
            # print("self.client_model_rank2")
            # print(self.client_model_rank2)
            self._build_client()
            if self.federated_config.rank > 0:
                self.logger.info(f"building client {self.federated_config.rank} ...")
                self.logger.info(f"local rank {self.federated_config.rank}'s client ids "
                                 f"is {list(self.data.train_dataloader_dict.keys())}")
            else:
                self.logger.info("building centralized training")

    def train(self):
        # TODO phase decides train or test
        if self.federated_config.rank == 0:
            self.logger.debug(f"Server Start ...")
            self.server_manger.run()
            self.on_server_end()

        elif self.federated_config.rank > 0:
            self.logger.debug(f"Sub-Server {self.federated_config.rank} Training Start ...")
            self.client_manager.run()
            self.on_client_end()

        else:
            self.logger.debug(f"Centralized Training Start ...")
            self.client_trainer.cen_train()
            self.on_client_end()

    def on_server_end(self):
        """on_server_end"""
        self.handler.test_on_server()
        global_metric_save(self.handler, self.training_config, self.logger)

    def on_client_end(self, *args):
        ...
