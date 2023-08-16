"""BaseClientTrainer for FedETuning"""

from abc import ABC
from typing import List
from thop import profile
from thop import clever_format

import torch
from transformers import get_linear_schedule_with_warmup, AdamW
from torch.optim.lr_scheduler import LambdaLR

from utils import registry
from utils import get_parameter_number
from fedlab.utils import MessageCode, SerializationTool
from fedlab.core.client.trainer import ClientTrainer
from fedlab.core.client.manager import PassiveClientManager
from fedlab.core.client.manager import ORDINARY_TRAINER, SERIAL_TRAINER
from fedlab.core.server.handler import Aggregators

# martinc
import transformers

class BaseClientTrainer(ClientTrainer, ABC):
    # def __init__(self, model, train_dataset, valid_dataset):
    def __init__(self, client_model_rank2, client_model_rank4, client_model_rank8, client_model_rank16, train_dataset, valid_dataset, train_total_step):

        # self._model = model
        self.client_model_rank2 = client_model_rank2
        self.client_model_rank4 = client_model_rank4
        self.client_model_rank8 = client_model_rank8
        self.client_model_rank16 = client_model_rank16
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

        self.train_total_step = train_total_step
        # self.last_training_step = -1
        

        """
        # martinc testing print
        check_id_list = list(range(0, 100))
        print("check_id_list")
        print(check_id_list)
        for check_idx in check_id_list:
            check_train_loader = self._get_dataloader(dataset=self.train_dataset, client_id=check_idx)
            for step, batch in enumerate(check_train_loader):
                check_inputs = {'input_ids': batch[0],
                                'attention_mask': batch[1],
                                'labels': batch[3]
                                }
        """

        delta_args = registry.get("delta_config")
        self.server_rank = delta_args["lora_r"]
        #martinc client model low rank distribution
        """
        rank 2: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61
        rank 4: 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73
        rank 8: 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99
        """

        
        if(self.server_rank >= 16):
            self.client_rank2_id_list =  [0,   1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61]
            self.client_rank4_id_list =  [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73]
            self.client_rank8_id_list =  [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86]
            self.client_rank16_id_list = [38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
        elif(self.server_rank >= 8):
            # rank2: 25% , rank4: 25% , rank8: 50%
            self.client_rank2_id_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61]
            self.client_rank4_id_list = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73]
            self.client_rank8_id_list = [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
            self.client_rank16_id_list = []
        
        
        """
        print("martinc only client_rank8_id_list")
        self.client_rank2_id_list = []
        self.client_rank4_id_list = []
        self.client_rank8_id_list = list(range(0, 100))
        self.client_rank16_id_list = []
        """

        """
        #
        print("martinc only client_rank16_id_list")
        self.client_rank2_id_list = []
        self.client_rank4_id_list = []
        self.client_rank8_id_list = []
        self.client_rank16_id_list = list(range(0, 100))
        """
        self._before_training()

    def _before_training(self):
        """before training function"""

        self.type = SERIAL_TRAINER  # represent serial trainer

        config = registry.get("config")
        self.model_config = config.M
        self.data_config = config.D
        self.training_config = config.T
        self.federated_config = config.F

        self.client_num = len(config.F.clients_id_list)
        self.device = config.training_config.device
        self.rank = config.federated_config.rank
        # martinc
        self.mix_round_threshold = self.federated_config.mix_round_threshold
        self.alternate_lora_training = self.federated_config.alternate_lora_training
        self.reset_client_lora_begin = self.federated_config.reset_client_lora_begin
        self.average_same_rank_client_model = self.federated_config.average_same_rank_client_model
        print("martinc average_same_rank_client_model---------------------------:", self.average_same_rank_client_model)

        self.param_list = []
        self.logger = registry.get("logger")

        self._build_metric()
        self._build_eval()

        # key: client idx, value: valid metric
        self.loc_best_metric = {}
        # key: client idx, value: test metric
        self.loc_test_metric = {}
        # key: client idx, value: serialized params
        self.loc_best_params = {}
        # local patient times
        self.loc_patient_times = 0
        # local early stop
        self.stop_early = False

        self.metric_name = self.metric.metric_name
        # self._model.to(self.device)


        # martinc
        self.train_warmup_steps = int(self.train_total_step  * self.training_config.warmup_ratio)


        self.client_model_rank2.to(self.device)
        self.client_model_rank4.to(self.device)
        self.client_model_rank8.to(self.device)
        self.client_model_rank16.to(self.device)

        if self.federated_config.rank == -1:
            self._calculate_model_computation()

        # build mapping dict for client choose which rank model
        self.rank_mapping_dict = {}
        for i in range(len(self.client_rank2_id_list)):
            clientidx = self.client_rank2_id_list[i]
            self.rank_mapping_dict[clientidx] = 2

        for i in range(len(self.client_rank4_id_list)):
            clientidx = self.client_rank4_id_list[i]
            self.rank_mapping_dict[clientidx] = 4

        for i in range(len(self.client_rank8_id_list)):
            clientidx = self.client_rank8_id_list[i]
            self.rank_mapping_dict[clientidx] = 8

        for i in range(len(self.client_rank16_id_list)):
            clientidx = self.client_rank16_id_list[i]
            self.rank_mapping_dict[clientidx] = 16

    def _calculate_model_computation(self):

        dummy_idx = list(self.train_dataset.keys())[0]
        train_loader = self._get_dataloader(dataset=self.train_dataset, client_id=dummy_idx)
        for step, batch in enumerate(train_loader):
            self._model.train()
            batch = tuple(t.to(self.device) for t in batch)

            macs, params = profile(self._model.backbone, inputs=(batch[0],))
            flops, params = clever_format([macs, params], "%.3f")
            self.logger.debug(f"Model Type: {self.model_config.model_type}, "
                              f"Tuning Type: {self.training_config.tuning_type}, "
                              f"Parameters: {get_parameter_number(self._model.backbone)}, "
                              f"FLOPs: {flops}")
            break

    @property
    def uplink_package(self):
        return self.param_list

    def _train_alone(self, idx: int, model_parameters: torch.Tensor, server_round: int, client_model, client_rank: int, last_training_step: int, *args, **kwargs):
        """local training for Client"""

        train_loader = self._get_dataloader(dataset=self.train_dataset, client_id=idx)
        # print("------------------------------------------------------------------")
        # print("idx:", idx, "client_model", client_model)
        
        if model_parameters is not None:
            SerializationTool.deserialize_model(client_model, model_parameters, client_rank, self.server_rank)

        # martinc
        # reset model lora A, lora B
        # self._model.backbone.reset_all_lora_parameters()
        if (server_round % self.mix_round_threshold == 0 and (self.reset_client_lora_begin != 0)):
            if (self.reset_client_lora_begin == 1):
                torch.manual_seed(self.training_config.seed)
                torch.cuda.manual_seed(self.training_config.seed)
                torch.cuda.manual_seed_all(self.training_config.seed)
            elif (self.reset_client_lora_begin == 2):
                torch.manual_seed(self.training_config.seed + idx)
                torch.cuda.manual_seed(self.training_config.seed + idx)
                torch.cuda.manual_seed_all(self.training_config.seed + idx)
            client_model.backbone.reset_all_lora_parameters()

        """
        # martinc
        # reset model lora A, lora B
        # self._model.backbone.reset_all_lora_parameters()
        if (server_round % self.mix_round_threshold == 0):
            client_model.backbone.reset_all_lora_parameters()
        """
        """
        # martinc test check
        for name, parameter in client_model.named_parameters():
            if (name == "backbone.base_model.model.roberta.encoder.layer.0.attention.self.query.lora_A.default.weight"):# backbone.base_model.model.roberta.encoder.layer.3.attention.self.query.weight
                print("before train client_model idx:", idx , " parameter:", parameter)
        """
        # build optimizer,scheduler,loss
        # optimizer, scheduler = self._build_optimizer(client_model, len(train_loader))
        optimizer, scheduler = self._build_optimizer(client_model, last_training_step)
        client_model, optimizer = self._mixed_train_model(client_model, optimizer)
        self._build_loss()


        if (self.alternate_lora_training):
            # if (server_round % 1 == 0):
            if (server_round % 2 == 0):
                freeze_grad_name = "lora_B"
                unfreeze_grad_name = "lora_A"
            elif (server_round % 2 == 1):
                freeze_grad_name = "lora_A"
                unfreeze_grad_name = "lora_B"
            for name, parameter in client_model.named_parameters():
                # print("client idx:", idx, " name:", name, " parameter.requires_grad:", parameter.requires_grad)
                if (freeze_grad_name in name):
                    parameter.requires_grad = False
                if (unfreeze_grad_name in name):
                    parameter.requires_grad = True

        # martinc
        print("before client idx:", idx, " scheduler.last_epoch", scheduler.last_epoch, " scheduler.get_lr()", scheduler.get_lr())

        for epoch in range(0, int(self.training_config.num_train_epochs)):
            self._on_epoch_begin()
            self._on_epoch(client_model, train_loader, optimizer, scheduler, idx)
            self._on_epoch_end(client_model, idx, client_rank)
            if self.federated_config.pson and self.stop_early:
                self.logger.critical(f"local stop early in {epoch}")
                break

        # martinc
        print("after client idx:", idx, " scheduler.last_epoch", scheduler.last_epoch, " scheduler.get_lr()", scheduler.get_lr())

        last_training_step = scheduler.last_epoch - 1
        return last_training_step
        # TODO return self.last_training_step back to server, and to average, then pass back to client

        """# original place
        if (server_round % self.mix_round_threshold == 0):
            # print("idx:", idx, " server_round:", server_round, " self.mix_round_threshold:", self.mix_round_threshold)
            # print("client idx:", idx, " self.mix_round_threshold:", self.mix_round_threshold, " server backbone.base_model.model.roberta.encoder.layer.3.attention.self.value.round_count.round_count")
            # print(int(self._model.state_dict()["backbone.base_model.model.roberta.encoder.layer.3.attention.self.value.round_count.round_count"].item()))
            # martinc
            # reset model lora A, lora B
            client_model.backbone.merge_lora_reuse()

            # martinc
            # reset model lora A, lora B
            client_model.backbone.reset_all_lora_parameters()
            # self._model.backbone.reset_zero_all_lora_parameters()
            # reset lora layer grad
            for name, parameter in client_model.named_parameters():
                if "lora" in name:
                    parameter.grad.zero_()
        """

        """
        # martinc test check
        for name, parameter in client_model.named_parameters():
            if (name == "backbone.base_model.model.roberta.encoder.layer.3.attention.self.query.weight"):# backbone.base_model.model.roberta.encoder.layer.3.attention.self.query.weight
                print("after train client_model idx:", idx , " parameter:", parameter)
        """

    def _get_dataloader(self, dataset, client_id: int):
        """Get :class:`DataLoader` for ``client_id``."""
        if isinstance(dataset, dict):
            data_loader = dataset[client_id]
        else:
            data_loader = dataset
        return data_loader

    def local_process(self, id_list: List, payload: List):
        """local process for Federated Learning"""
        model_parameters = payload[0]
        # print("local_process model_parameters------------------------------------")
        # print(model_parameters)
        self.param_list = self.fed_train(model_parameters, id_list)
        return self.param_list

    def average_same_rank_model(self, rank_param_list: List, client_model, client_rank, server_round):# rank2_param_list, self.client_model_rank2, 2
        rank_serialized_parameters = Aggregators.fedavg_aggregate(rank_param_list)
        # SerializationTool.deserialize_model(client_model, rank_serialized_parameters, self.server_rank, self.server_rank)
        SerializationTool.original_deserialize_model(client_model, rank_serialized_parameters)
        if (server_round % self.mix_round_threshold == 0):
            # reset model lora A, lora B
            client_model.backbone.merge_lora_reuse()
            # reset model lora A, lora B
            client_model.backbone.reset_all_lora_parameters()
            # reset lora layer grad
            for name, parameter in client_model.named_parameters():
                if "lora" in name:
                    parameter.grad.zero_()
        return self.model_parameters(client_model, client_rank, self.server_rank)

    def fed_train(self, model_parameters: torch.Tensor, id_list: List):
        param_list = []
        # server_round = int(model_parameters[-1].item())
        # model_parameters = model_parameters[:-1]

        # return [torch.cat((self.model_parameters(self._model, self.server_rank, self.server_rank), torch.tensor([self.round + 1]), torch.tensor([self.last_training_step])), dim=0)]
        server_last_training_step = int(model_parameters[-1].item()) # average by server
        server_round = int(model_parameters[-2].item())
        model_parameters = model_parameters[:-2]
        # print("client fed_train self.last_training_step:", self.last_training_step)

        # average_same_rank_client_model parameter
        rank2_param_list = []
        rank4_param_list = []
        rank8_param_list = []
        rank16_param_list = []

        rank2_correct_list = []
        rank2_total_list = []
        rank2_loss_list = []
        # self.last_training_step
        rank2_last_training_step_list = []

        rank4_correct_list = []
        rank4_total_list = []
        rank4_loss_list = []
        rank4_last_training_step_list = []

        rank8_correct_list = []
        rank8_total_list = []
        rank8_loss_list = []
        rank8_last_training_step_list = []

        rank16_correct_list = []
        rank16_total_list = []
        rank16_loss_list = []
        rank16_last_training_step_list = []

        # self.tr_loss/self.global_step
        
        # different idx has different rank client model
        """
        client
        rank 2: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61
        rank 4: 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73
        rank 8: 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99
        self.client_model_rank2 = client_model_rank2
        self.client_model_rank4 = client_model_rank4
        self.client_model_rank8 = client_model_rank8
        """
        for idx in id_list:
            # return [torch.cat((self.model_parameters(self._model, self.server_rank, self.server_rank), torch.tensor([self.round + 1]), torch.tensor([self.last_training_step])), dim=0)]
            last_training_step = server_last_training_step # average by server
            print("client idx:", idx, " client fed_train last_training_step:", last_training_step)

            client_rank = self.rank_mapping_dict[idx]
            if(client_rank == 2):
                rank_model = self.client_model_rank2
            elif(client_rank == 4):
                rank_model = self.client_model_rank4
            elif(client_rank == 8):
                rank_model = self.client_model_rank8
            elif(client_rank == 16):
                rank_model = self.client_model_rank16


            last_training_step = self._train_alone(
                idx=idx,
                model_parameters=model_parameters,
                server_round=server_round,
                client_model=rank_model,
                client_rank=client_rank,
                last_training_step=last_training_step
            )
            # original open
            # param_list.append(self.model_parameters)
            # print("client_rank:", client_rank)

            if (self.average_same_rank_client_model):
                if (client_rank == 2):
                    # rank2_param_list.append(self.model_parameters(rank_model, client_rank, self.server_rank))
                    rank2_param_list.append(SerializationTool.original_serialize_model(rank_model))
                    rank2_last_training_step_list.append(last_training_step)
                    """
                    rank2_correct_list.append(self.correct)
                    rank2_total_list.append(self.total)
                    rank2_loss_list.append(self.tr_loss/self.global_step)
                    """
                elif (client_rank == 4):
                    # rank4_param_list.append(self.model_parameters(rank_model, client_rank, self.server_rank))
                    rank4_param_list.append(SerializationTool.original_serialize_model(rank_model))
                    rank4_last_training_step_list.append(last_training_step)
                    """
                    rank4_correct_list.append(self.correct)
                    rank4_total_list.append(self.total)
                    rank4_loss_list.append(self.tr_loss/self.global_step)
                    """
                elif (client_rank == 8):
                    # rank8_param_list.append(self.model_parameters(rank_model, client_rank, self.server_rank))
                    rank8_param_list.append(SerializationTool.original_serialize_model(rank_model))
                    rank8_last_training_step_list.append(last_training_step)
                    """
                    rank8_correct_list.append(self.correct)
                    rank8_total_list.append(self.total)
                    rank8_loss_list.append(self.tr_loss/self.global_step)
                    """
                elif (client_rank == 16):
                    # rank16_param_list.append(self.model_parameters(rank_model, client_rank, self.server_rank))
                    rank16_param_list.append(SerializationTool.original_serialize_model(rank_model))
                    rank16_last_training_step_list.append(last_training_step)
                    """
                    rank16_correct_list.append(self.correct)
                    rank16_total_list.append(self.total)
                    rank16_loss_list.append(self.tr_loss/self.global_step)
                    """
            else:
                if (server_round % self.mix_round_threshold == 0):
                    # print("idx:", idx, " server_round:", server_round, " self.mix_round_threshold:", self.mix_round_threshold)
                    # print("client idx:", idx, " self.mix_round_threshold:", self.mix_round_threshold, " server backbone.base_model.model.roberta.encoder.layer.3.attention.self.value.round_count.round_count")
                    # print(int(self._model.state_dict()["backbone.base_model.model.roberta.encoder.layer.3.attention.self.value.round_count.round_count"].item()))
                    # martinc
                    # reset model lora A, lora B
                    rank_model.backbone.merge_lora_reuse()

                    # martinc
                    # reset model lora A, lora B
                    rank_model.backbone.reset_all_lora_parameters()
                    # self._model.backbone.reset_zero_all_lora_parameters()
                    # reset lora layer grad
                    for name, parameter in rank_model.named_parameters():
                        if "lora" in name:
                            parameter.grad.zero_()

                """
                # martinc test check
                for name, parameter in rank_model.named_parameters():
                    if (name == "backbone.base_model.model.roberta.encoder.layer.3.attention.self.query.weight"):# backbone.base_model.model.roberta.encoder.layer.3.attention.self.query.weight
                        print("after train rank_model idx:", idx , " parameter:", parameter)
                """
                # original
                # param_list.append(self.model_parameters(rank_model, client_rank, self.server_rank))
                # self.last_training_step
                param_list.append(torch.cat((self.model_parameters(rank_model, client_rank, self.server_rank), torch.tensor([last_training_step])),  dim=0))
                # param_list.append(torch.cat((self.model_parameters(rank_model, client_rank, self.server_rank), torch.tensor([self.correct]), torch.tensor([self.total]), torch.tensor([(self.tr_loss/self.global_step)])), dim=0))
                # param_list.append(torch.cat(self.model_parameters(rank_model, client_rank, self.server_rank), torch.tensor([self.correct]), torch.tensor([self.total]), torch.tensor([(self.tr_loss/self.global_step)])), dim=0)

        if (self.average_same_rank_client_model):
            if (len(rank2_param_list) > 0):
                averaged_serialized_parameters = self.average_same_rank_model(rank2_param_list, self.client_model_rank2, 2, server_round) # rank_param_list: List, client_model, client_rank, server_round)
                param_list.append(torch.cat((averaged_serialized_parameters, torch.tensor([sum(rank2_last_training_step_list)/len(rank2_last_training_step_list)])), dim=0))
                # param_list.append(torch.cat((averaged_serialized_parameters, torch.tensor([sum(rank2_correct_list)]), torch.tensor([sum(rank2_total_list)]), torch.tensor([sum(rank2_loss_list)/len(rank2_loss_list)])), dim=0))
                # param_list.append(averaged_serialized_parameters)
            if (len(rank4_param_list) > 0):
                averaged_serialized_parameters = self.average_same_rank_model(rank4_param_list, self.client_model_rank4, 4, server_round) # rank_param_list: List, client_model, client_rank, server_round)
                param_list.append(torch.cat((averaged_serialized_parameters, torch.tensor([sum(rank4_last_training_step_list)/len(rank4_last_training_step_list)])), dim=0))
                # param_list.append(torch.cat((averaged_serialized_parameters, torch.tensor([sum(rank4_correct_list)]), torch.tensor([sum(rank4_total_list)]), torch.tensor([sum(rank4_loss_list)/len(rank4_loss_list)])), dim=0))
                # param_list.append(averaged_serialized_parameters)
            if (len(rank8_param_list) > 0):
                averaged_serialized_parameters = self.average_same_rank_model(rank8_param_list, self.client_model_rank8, 8, server_round) # rank_param_list: List, client_model, client_rank, server_round)
                param_list.append(torch.cat((averaged_serialized_parameters, torch.tensor([sum(rank8_last_training_step_list)/len(rank8_last_training_step_list)])), dim=0))
                # param_list.append(torch.cat((averaged_serialized_parameters, torch.tensor([sum(rank8_correct_list)]), torch.tensor([sum(rank8_total_list)]), torch.tensor([sum(rank8_loss_list)/len(rank8_loss_list)])), dim=0))
                # param_list.append(averaged_serialized_parameters)
            if (len(rank16_param_list) > 0):
                averaged_serialized_parameters = self.average_same_rank_model(rank16_param_list, self.client_model_rank16, 16, server_round) # rank_param_list: List, client_model, client_rank, server_round)
                param_list.append(torch.cat((averaged_serialized_parameters, torch.tensor([sum(rank16_last_training_step_list)/len(rank16_last_training_step_list)])), dim=0))
                # param_list.append(torch.cat((averaged_serialized_parameters, torch.tensor([sum(rank16_correct_list)]), torch.tensor([sum(rank16_total_list)]), torch.tensor([sum(rank16_loss_list)/len(rank16_loss_list)])), dim=0))
                # param_list.append(averaged_serialized_parameters)
                # print("done rank16_param_list average_same_rank_model")

        """
        print("fed_train before return param_list")
        print("param_list")
        print(param_list)
        print("len(param_list):", len(param_list))
        print("param_list[0].size():", param_list[0].size())
        print("param_list[1].size():", param_list[1].size())
        """
        return param_list

    def cen_train(self, *args):
        self._train_alone(
            idx=-1,
            model_parameters=None,
        )

    # Local Training Functions
    def _build_loss(self):
        self.criterion = registry.get_loss_class(self.training_config.loss_name)(
            config=self.training_config
        )

    def get_linear_start_end_ratio_schedule(self, optimizer, num_training_steps, start_lr, end_lr, last_epoch=-1):# (self, optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
        """
        Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0,
        after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

        Args:
            optimizer (:class:`~torch.optim.Optimizer`):
                The optimizer for which to schedule the learning rate.
            num_warmup_steps (:obj:`int`):
                The number of steps for the warmup phase.
            num_training_steps (:obj:`int`):
                The totale number of training steps.
            last_epoch (:obj:`int`, `optional`, defaults to -1):
                The index of the last epoch when resuming training.

        Return:
            :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
        """

        def lr_lambda(current_step: int):
            # if current_step < num_warmup_steps:
            #     return float(current_step) / float(max(1, num_warmup_steps))
            # print("start_lr:", start_lr, " end_lr:", end_lr, " num_training_steps:", num_training_steps, " current_step:", current_step)
            # new_lr = start_lr - (((start_lr - end_lr) / float(num_training_steps)) *  float(current_step))
            # print("new_lr:", new_lr, " start_lr:", start_lr, " end_lr:", end_lr, " num_training_steps:", num_training_steps, " current_step:", current_step)
            
            mul_factor = 1 - (((start_lr - end_lr) * current_step) / (num_training_steps * start_lr))

            # return mul_factor
            return max(
                0.0, mul_factor
            )
            
            """
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )
            """
        return LambdaLR(optimizer, lr_lambda, last_epoch)

    def _build_optimizer(self, model, last_training_step):
        """
        if self.training_config.max_steps > 0:
            t_total = self.training_config.max_steps
            self.training_config.num_train_epochs = \
                self.training_config.max_steps // (train_dl_len // self.training_config.gradient_accumulation_steps) + 1
        else:
            t_total = \
                train_dl_len // self.training_config.gradient_accumulation_steps * self.training_config.num_train_epochs
        """

        

        # Prepare optimizer and schedule (linear warmup and decay)
        optimizer_grouped_parameters = self.get_optimized_model_params(model)

        optimizer = AdamW(
            # [{'params': optimizer_grouped_parameters, 'initial_lr': self.training_config.learning_rate}],
            optimizer_grouped_parameters,
            lr=self.training_config.learning_rate,
            eps=self.training_config.adam_epsilon
        )

        #original
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.train_warmup_steps,
            num_training_steps=self.train_total_step, last_epoch=last_training_step
        )
        
        """
        # original
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=self.training_config.learning_rate,
            eps=self.training_config.adam_epsilon
        )

        #original
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.training_config.warmup_steps,
            num_training_steps=t_total
        )
        """

        """
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=start_learning_rate,
            eps=self.training_config.adam_epsilon
        )


        # def get_linear_start_end_ratio_schedule(self, optimizer, num_training_steps, start_lr, end_lr, last_epoch=-1):
        scheduler = self.get_linear_start_end_ratio_schedule(
            optimizer, num_training_steps=t_total, start_lr=start_learning_rate, end_lr=end_learning_rate
        )
        """
        return optimizer, scheduler

    def get_optimized_model_params(self, model):
        # Prepare optimizer and schedule (linear warmup and decay)
        """
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.backbone.named_parameters() if
                        not any(nd in n for nd in no_decay)], 'weight_decay': self.training_config.weight_decay},
            {'params': [p for n, p in model.backbone.named_parameters() if
                        any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]
        """
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.backbone.named_parameters() if
                        not any(nd in n for nd in no_decay)], 'weight_decay': self.training_config.weight_decay, 'initial_lr': self.training_config.learning_rate},
            {'params': [p for n, p in model.backbone.named_parameters() if
                        any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'initial_lr': self.training_config.learning_rate},
        ]
        # Both pieces of code have the same effect
        # optimizer_grouped_parameters = [
        #     {"params": filter(lambda x: x.requires_grad, model.bert.parameters()),
        #      'weight_decay': 0.0},
        # ]

        return optimizer_grouped_parameters

    def _mixed_train_model(self, model, optimizer):
        if self.training_config.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.training_config.fp16_opt_level)

            # multi-gpu training (should be after apex fp16 initialization)
        if self.training_config.n_gpu > 1:
            self.logger.warning("We haven't tested our model under multi-gpu. Please be aware!")
            model = torch.nn.DataParallel(model)

        return model, optimizer

    # Local Test Function
    def _build_metric(self):
        self.metric = registry.get_metric_class(self.training_config.metric_name)(
            self.data_config.task_name, self.training_config.is_decreased_valid_metric
        )

    def _build_eval(self):
        self.eval = registry.get_eval_class(self.training_config.metric_name)(
            self.device, self.metric
        )

    def test_on_client(self, test_dataloader, client_rank):

        for idx in self.loc_best_params:
            loc_best_params = self.loc_best_params[idx]
            SerializationTool.deserialize_model(self._model, loc_best_params, client_rank, self.server_rank)
            result = self.eval.test_and_eval(
                model=self._model,
                valid_dl=test_dataloader,
                model_type=self.model_config.model_type,
                model_output_mode=self.model_config.model_output_mode
            )
            test_metric, test_loss = result[self.metric_name], result["eval_loss"]
            self.logger.critical(
                f"{self.data_config.task_name.upper()} Test, "
                f"Client:{idx}, Test loss:{test_loss:.3f}, "
                f"Test {self.metric_name}:{test_metric:.3f}"
            )
            self.loc_test_metric[idx] = test_metric

    # Local Epoch Function
    def _on_epoch_begin(self):
        self.global_step = 0
        self.tr_loss, self.logging_loss = 0.0, 0.0
        self.total, self.correct = 0, 0

    def _on_epoch(self, client_model, train_loader, optimizer, scheduler, idx):
        for step, batch in enumerate(train_loader):
            # print("idx:", idx, " step:", step, " client scheduler.get_last_lr():", scheduler.get_last_lr())
            # print("idx:", idx, " step:", step, " client optimizer.param_groups[0]['lr']:", optimizer.param_groups[0]['lr'])
            
            # if step >= 2:
            #     break
            client_model.train()
            batch = tuple(t.to(self.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[3]
                      }
            label = inputs['labels']
            if self.model_config.model_type != 'distilbert' or self.model_config.model_type != 'roberta':
                # XLM, DistilBERT and RoBERTa don't use segment_ids
                inputs['token_type_ids'] = batch[2] \
                    if self.model_config.model_type in ['bert', 'xlnet'] else None
            outputs = client_model(inputs)

            loss, logits = outputs[:2]
            _, predicted = torch.max(logits, 1)

            optimizer.zero_grad()
            if self.training_config.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if self.training_config.gradient_accumulation_steps > 1:
                loss = loss / self.training_config.gradient_accumulation_steps

            if self.training_config.fp16:
                try:
                    from apex import amp
                except ImportError:
                    raise ImportError(
                        "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            self.tr_loss += loss.item()
            if (step + 1) % self.training_config.gradient_accumulation_steps == 0:
                if self.training_config.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.training_config.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(client_model.parameters(), self.training_config.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule

                self.global_step += 1

            self.total += label.size(0)
            if self.model_config.model_output_mode == "seq_classification":
                self.correct += (predicted == label).sum().item()

    def _on_epoch_end(self, client_model, idx, client_rank):
        """on epoch end"""
        """
        self.logger.info(f"{self.data_config.task_name.upper()} Train, "
                         f"Client:{idx}, Loss:{self.tr_loss/self.global_step:.3f}, "
                         f"Accuracy:{self.correct/self.total:.3f}")
        """
        self.logger.info(f"{self.data_config.task_name.upper()} Train, "
                         f"Client:{idx}, client_rank:{client_rank}, Loss:{self.tr_loss/self.global_step:.3f}, "
                         f"Accuracy:{self.correct/self.total:.3f}")

        if not self.federated_config.pson:
            # not need for local test
            return

        valid_data = self._get_dataloader(dataset=self.valid_dataset, client_id=idx)

        result = self.eval.test_and_eval(
            model=client_model,
            valid_dl=valid_data,
            model_type=self.model_config.model_type,
            model_output_mode=self.model_config.model_output_mode
        )

        test_metric, test_loss = result[self.metric_name], result["eval_loss"]

        # TODO hard code
        if not self.loc_best_metric.get(idx, None):
            self.loc_best_metric[idx] = float('-inf')
        if self.loc_best_metric[idx] < test_metric:
            self.loc_best_metric[idx] = test_metric
            self.loc_best_params[idx] = SerializationTool.serialize_model(client_model, client_rank, self.server_rank)
            self.loc_patient_times = 0
        else:
            self.loc_patient_times += 1

        self.logger.debug(f"{self.data_config.task_name.upper()} Eval, "
                          f"Client:{idx}, Loss:{test_loss:.3f}, "
                          f"Current {self.metric_name}:{test_metric:.3f}, "
                          f"Best {self.metric_name}:{self.loc_best_metric[idx]:.3f}")

        if self.loc_patient_times >= self.training_config.patient_times:
            self.stop_early = True


class BaseClientManager(PassiveClientManager, ABC):
    def __init__(self, network, trainer):
        self.logger = registry.get("logger")
        super().__init__(network, trainer, self.logger)

    def main_loop(self):
        """Actions to perform when receiving a new message, including local trainers.

        Main procedure of each client:
            1. client waits for data from server (PASSIVELY).
            2. after receiving data, client start local model trainers procedure.
            3. client synchronizes with server actively.
        """
        while True:
            sender_rank, message_code, payload = self._network.recv(src=0)

            if message_code == MessageCode.Exit:
                # client exit feedback
                if self._network.rank == self._network.world_size - 1:
                    self._network.send(message_code=MessageCode.Exit, dst=0)
                break

            elif message_code == MessageCode.ParameterUpdate:

                id_list, payload = payload[0].to(
                    torch.int32).tolist(), payload[1:]

                # check the trainer type
                if self._trainer.type == SERIAL_TRAINER:  # serial
                    self._trainer.local_process(
                        id_list=id_list,
                        payload=payload
                    )

                elif self._trainer.type == ORDINARY_TRAINER:  # ordinary
                    assert len(id_list) == 1
                    self._trainer.local_process(payload=payload)

                self.synchronize()

            else:
                raise ValueError(f"Invalid MessageCode {message_code}. Please check MessageCode list.")

    def synchronize(self):
        """Synchronize with server"""
        self.logger.info("Uploading information to server.")
        self._network.send(
            content=self._trainer.uplink_package,
            message_code=MessageCode.ParameterUpdate,
            dst=0
        )
