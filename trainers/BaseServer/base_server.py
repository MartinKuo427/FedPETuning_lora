""" BaseServer for FedETuning """

import os
import random
import threading
from abc import ABC

import torch

from utils.register import registry

from fedlab.core.server.handler import Aggregators
from fedlab.core.server.handler import ParameterServerBackendHandler
from fedlab.core.server.manager import ServerManager
from fedlab.utils.serialization import SerializationTool
from fedlab.utils import MessageCode
from fedlab.core.coordinator import Coordinator

import traceback

from torch.utils.tensorboard import SummaryWriter

import numpy as np

class BaseSyncServerHandler(ParameterServerBackendHandler, ABC):
    def __init__(self, model, valid_data, test_data):

        self.valid_data = valid_data
        self.test_data = test_data

        config = registry.get("config")
        self.model_config = config.model_config
        self.data_config = config.data_config
        self.training_config = config.training_config
        self.federated_config = config.federated_config

        self.logger = registry.get("logger")

        self.device = config.training_config.device
        self._model = model.to(self.device)

        # basic setting
        self.client_num_in_total = config.federated_config.clients_num
        self.sample_ratio = config.federated_config.sample

        # martinc server_rank
        delta_args = registry.get("delta_config")
        self.server_rank = delta_args["lora_r"]

        # client buffer
        self.client_buffer_cache = []
        self.cache_cnt = 0

        # stop condition
        self.global_round = config.federated_config.rounds
        self.round = 0

        print("martinc self.global_round:", self.global_round)

        #  metrics & eval
        self._build_metric()
        self._build_eval()
        self.global_valid_best_metric = \
            float("inf") if self.training_config.is_decreased_valid_metric else -float("inf")
        self.global_test_best_metric = 0.0
        self.metric_log = {
            "model_type": self.model_config.model_type,
            "clients_num": self.federated_config.clients_num,
            "alpha": self.federated_config.alpha, "task": self.data_config.task_name,
            "fl_algorithm": self.federated_config.fl_algorithm,
            "info": f"{self.model_config.model_type}_{self.federated_config.fl_algorithm}_"
                    f"{self.federated_config.clients_num}_{self.federated_config.alpha}",
            "logs": []
        }
        # metric line
        self.metric_name = self.metric.metric_name
        times = registry.get("run_time")
        if self.training_config.do_grid:
            grid_info = registry.get("grid_info")
            self.metric_line = f"{times}_{self.model_config.model_type}_{self.training_config.tuning_type}_" \
                               f"seed={self.training_config.seed}_rounds={self.federated_config.rounds}_" \
                               f"cli={self.federated_config.clients_num}_alp={self.federated_config.alpha}_" \
                               f"sap={self.federated_config.sample}_epo={self.training_config.num_train_epochs}_" \
                               f"lr={self.training_config.learning_rate}_{grid_info}_"
        else:
            self.metric_line = f"{times}_{self.model_config.model_type}_{self.training_config.tuning_type}_" \
                               f"seed={self.training_config.seed}_rounds={self.federated_config.rounds}_" \
                               f"cli={self.federated_config.clients_num}_alp={self.federated_config.alpha}_" \
                               f"sap={self.federated_config.sample}_lr={self.training_config.learning_rate}_" \
                               f"epo={self.training_config.num_train_epochs}_"
        # global model
        self.glo_save_file = os.path.join(
            self.training_config.checkpoint_dir, f"{times}_{self.model_config.model_type}.pth")
        self.best_glo_params = None

        # martinc sender_rank_list: determine fed aggregation timing
        self.sender_rank_list = []
        self.world_size = self.federated_config.world_size

        # martinc global learning rate scheduler
        # self.global_start_learning_rate = self.training_config.learning_rate
        # self.global_learning_rate_array = np.linspace(self.training_config.learning_rate, 0.0, num=(self.global_round + 1))

        # martinc last epoch
        self.last_training_step = -1

        # martinc tensorboard writer
        self.writer = SummaryWriter()

        self.train_correct_list = []
        self.train_total_list = []
        self.train_loss_list = []

        """
        # param_list.append(torch.cat(self.model_parameters(rank_model, client_rank, self.server_rank), self.correct, self.total, (self.tr_loss/self.global_step)), dim=0)

        for index in range(len(model_parameters_list)):
            train_loss = model_parameters_list[index][-1]
            model_parameters_list[index] = model_parameters_list[index][:-3]
        """
    def _build_eval(self):
        self.eval = registry.get_eval_class(self.training_config.metric_name)(
            self.device, self.metric
        )

    def _build_metric(self):
        self.metric = registry.get_metric_class(self.training_config.metric_name)(
            self.data_config.task_name, self.training_config.is_decreased_valid_metric
        )

    def stop_condition(self) -> bool:
        return self.round >= self.global_round

    def sample_clients(self):
        selection = random.sample(
            range(self.client_num_in_total),
            self.client_num_per_round
        )
        return selection

    def _update_global_model(self, payload, sender_rank):
        assert len(payload) > 0

        if len(payload) == 1:
            self.client_buffer_cache.append(payload[0].clone())
        else:
            self.client_buffer_cache += payload  # serial trainer

        assert len(self.client_buffer_cache) <= self.client_num_per_round

        # martinc
        self.sender_rank_list.append(sender_rank)
        # if len(self.client_buffer_cache) == self.client_num_per_round:
        if (len(self.sender_rank_list) == (self.world_size - 1)):
            model_parameters_list = self.client_buffer_cache

            # martinc decompose message
            # param_list.append(torch.cat(self.model_parameters(rank_model, client_rank, self.server_rank), self.correct, self.total, (self.tr_loss/self.global_step)), dim=0)
            # self.train_correct_list = []
            # self.train_total_list = []
            # self.train_loss_list = []
            """
            for index in range(len(model_parameters_list)):
                train_loss = model_parameters_list[index][-1]
                train_total = model_parameters_list[index][-2]
                train_correct = model_parameters_list[index][-3]
                self.train_loss_list.append(train_loss.detach())
                self.train_total_list.append(train_total.item())
                self.train_correct_list.append(train_correct.item())
                model_parameters_list[index] = model_parameters_list[index][:-3]
            """

            last_training_step_list = []
            for index in range(len(model_parameters_list)):
                temp_last_training_step = model_parameters_list[index][-1]
                last_training_step_list.append(temp_last_training_step)
                model_parameters_list[index] = model_parameters_list[index][:-1]
            
            self.last_training_step = int(sum(last_training_step_list) / len(last_training_step_list))

            # martinc
            print("server average self.last_training_step:", self.last_training_step)

            self.logger.debug(
                f"Model parameters aggregation, number of aggregation elements {len(model_parameters_list)}"
            )

            # use aggregator
            serialized_parameters = Aggregators.fedavg_aggregate(model_parameters_list)
            SerializationTool.deserialize_model(self._model, serialized_parameters, self.server_rank, self.server_rank)
            """
            # martinc
            print("server backbone.base_model.model.roberta.encoder.layer.3.attention.self.value.round_count.round_count")
            print(self._model.state_dict()["backbone.base_model.model.roberta.encoder.layer.3.attention.self.value.round_count.round_count"])
            self._model.state_dict()["backbone.base_model.model.roberta.encoder.layer.3.attention.self.value.round_count.round_count"].add_(1)
            """

            self.round += 1

            self.valid_on_server()

            if self.federated_config.test_rounds:
                if self.round % self.federated_config.log_test_len == 0:
                    result = self.test_on_server()
                    if "test_rounds" not in self.metric_log:
                        self.metric_log["test_rounds"] = {}
                    self.metric_log["test_rounds"][f"round_{self.round}"] \
                        = result[self.metric_name]
            
            """
            if (self.round > 5):
                print("martinc check quit-----------")
                quit()
            """
            """
            print("server self.round:", self.round)
            print("server self.model_parameters")
            print(self.model_parameters)
            print("server self.model_parameters.size():", self.model_parameters.size())
            """
            """
            if (self.round > 5):
                print("martinc check quit-----------")
                quit()
            """
            """
            # martinc test log
            print("self._model")
            print(self._model)
            for name, parameter in self._model.named_parameters():
                print("name:-------------------------", name)
                print("parameter.size():", parameter.size())
            """
            # print("self._model.state_dict()")
            # print(self._model.state_dict())# .state_dict()
            # backbone.base_model.model.roberta.encoder.layer.3.attention.self.value.round_count.round_count

            # torch.add(self._model.state_dict()["backbone.base_model.model.roberta.encoder.layer.3.attention.self.value.round_count.round_count"], 1)
            # print("after add backbone.base_model.model.roberta.encoder.layer.3.attention.self.value.round_count.round_count")
            # print(self._model.state_dict()["backbone.base_model.model.roberta.encoder.layer.3.attention.self.value.round_count.round_count"])

            """
            if (self.round > 5):
                print("martinc check quit-----------")
                quit()
            """
            # reset cache cnt
            self.client_buffer_cache = []

            # reset self.sender_rank_list
            self.sender_rank_list = []

            return True  # return True to end this round.
        else:
            return False

    @property
    def client_num_per_round(self):
        return max(1, int(self.sample_ratio * self.client_num_in_total))

    @property
    def downlink_package(self):
        """Property for manager layer. BaseServer manager will call this property when activates clients."""
        # original
        # return [self.model_parameters]
        # self.last_training_step = 100
        print("martinc server downlink_package self.last_training_step:", self.last_training_step)
        return [torch.cat((self.model_parameters(self._model, self.server_rank, self.server_rank), torch.tensor([self.round + 1]), torch.tensor([self.last_training_step])), dim=0)]
        # return [torch.cat((self.model_parameters(self._model, self.server_rank, self.server_rank), torch.tensor([self.round + 1]), torch.tensor([self.global_learning_rate_array[min(len(self.global_learning_rate_array)-1, self.round)]]), torch.tensor([self.global_learning_rate_array[min(len(self.global_learning_rate_array)-1, self.round+1)]])), dim=0)]
        # return [torch.cat((cat_t, torch.tensor([self.round + 1])), dim=0)]
    # self.model_parameters(rank_model, client_rank)

    @property
    def if_stop(self):
        """
        class:`NetworkManager` keeps monitoring this attribute,
        and it will stop all related processes and threads when ``True`` returned.
        """
        return self.round >= self.global_round

    def valid_on_server(self):

        result = self.eval.test_and_eval(
            model=self._model,
            valid_dl=self.valid_data,
            model_type=self.model_config.model_type,
            model_output_mode=self.model_config.model_output_mode
        )

        self.on_round_end(result)

    def test_on_server(self):

        SerializationTool.deserialize_model(self._model, self.best_glo_params, self.server_rank, self.server_rank)

        result = self.eval.test_and_eval(
            model=self._model,
            valid_dl=self.test_data,
            model_type=self.model_config.model_type,
            model_output_mode=self.model_config.model_output_mode
        )

        self.logger.critical(f"task:{self.data_config.task_name}, Setting:{self.metric_log['info']}, "
                             f"Test {self.metric_name.upper()}:{result[self.metric_name]:.3f}")

        self.global_test_best_metric = result[self.metric_name]

        return result

    def save_global_model(self):
        torch.save(self.best_glo_params, self.glo_save_file)

    def on_round_end(self, result):
        test_metric, test_loss = result[self.metric_name], result["eval_loss"]

        """
        train_mean_loss_vals = torch.stack(self.train_loss_list).mean().item()
        train_movavg = sum(self.train_correct_list) / sum(self.train_total_list)        

        self.writer.add_scalar('train_loss/round', train_mean_loss_vals, self.round)
        self.writer.add_scalar('train_metric:' + self.metric_name + '/round', train_movavg, self.round)
        """
        self.writer.add_scalar('eval_loss/round', test_loss, self.round)
        self.writer.add_scalar('eval_metric/round', test_metric, self.round)
        # self.writer.add_scalar('lr/train', current_lr, step)
        self.writer.flush()

        # TODO hard code
        if self.global_valid_best_metric < test_metric:
            self.global_valid_best_metric = test_metric
            self.best_glo_params = SerializationTool.serialize_model(self._model, self.server_rank, self.server_rank)

        self.logger.info(f"{self.data_config.task_name}-{self.model_config.model_type} "
                         f"train with client={self.federated_config.clients_num}_"
                         f"alpha={self.federated_config.alpha}_"
                         f"epoch={self.training_config.num_train_epochs}_"
                         f"seed={self.training_config.seed}_"
                         f"comm_round={self.federated_config.rounds}")

        self.logger.debug(f"{self.federated_config.fl_algorithm} Eval "
                          f"Round:{self.round}, Loss:{test_loss:.3f}, "
                          f"Current {self.metric_name}:{test_metric:.3f}, "
                          f"Best {self.metric_name}:{self.global_valid_best_metric:.3f}")

        self.metric_log["logs"].append(
            {f"round_{self.round}": {
                "loss": f"{test_loss:.3f}",
                f"{self.metric.metric_name}": f"{test_metric:.3f}"
            }
            }
        )


class BaseServerManager(ServerManager):
    """Synchronous communication

    BaseServerManager.run()
    setup() main_loop() shut_down()

    """

    def __init__(self, network, handler):
        super(BaseServerManager, self).__init__(network, handler)

        self.logger = registry.get("logger")

    def setup(self):
        self._network.init_network_connection()

        rank_client_id_map = {}

        for rank in range(1, self._network.world_size):
            _, _, content = self._network.recv(src=rank)
            rank_client_id_map[rank] = content[0].item()
        self.coordinator = Coordinator(rank_client_id_map, mode='GLOBAL')  # mode='GLOBAL'
        if self._handler is not None:
            self._handler.client_num_in_total = self.coordinator.total

    def main_loop(self):

        while self._handler.if_stop is not True:
            activate = threading.Thread(target=self.activate_clients)
            activate.start()

            while True:
                sender_rank, message_code, payload = self._network.recv()

                if message_code == MessageCode.ParameterUpdate:
                    if self._handler._update_global_model(payload, sender_rank):
                        break
                else:
                    raise Exception(
                        "Unexpected message code {}".format(message_code))

    def shutdown(self):
        """Shutdown stage."""
        self.shutdown_clients()
        super().shutdown()

    def activate_clients(self):

        self.logger.info("BaseClient activation procedure")
        clients_this_round = self._handler.sample_clients()
        rank_dict = self.coordinator.map_id_list(clients_this_round)

        self.logger.info("BaseClient id list: {}".format(clients_this_round))

        for rank, values in rank_dict.items():
            downlink_package = self._handler.downlink_package
            id_list = torch.Tensor(values).to(downlink_package[0].dtype)
            self._network.send(
                content=[id_list] + downlink_package,
                message_code=MessageCode.ParameterUpdate,
                dst=rank
            )

    def shutdown_clients(self):
        """Shutdown all clients.

        Send package to each client with :attr:`MessageCode.Exit`.

        Note:
            Communication agreements related: User can overwrite this function to define package
            for exiting information.
        """
        client_list = range(self._handler.client_num_in_total)
        rank_dict = self.coordinator.map_id_list(client_list)

        for rank, values in rank_dict.items():
            downlink_package = self._handler.downlink_package
            id_list = torch.Tensor(values).to(downlink_package[0].dtype)
            self._network.send(content=[id_list] + downlink_package,
                               message_code=MessageCode.Exit,
                               dst=rank)

        # wait for client exit feedback
        _, message_code, _ = self._network.recv(
            src=self._network.world_size - 1
        )
        assert message_code == MessageCode.Exit
