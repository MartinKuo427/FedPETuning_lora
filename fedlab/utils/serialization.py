# Copyright 2021 Peng Cheng Laboratory (http://www.szpclab.com/) and FedLab Authors (smilelab.group)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch


class SerializationTool(object):
    @staticmethod
    def serialize_model_gradients(model: torch.nn.Module) -> torch.Tensor:
        gradients = [param.grad.data.view(-1) for param in model.parameters()]
        m_gradients = torch.cat(gradients)
        m_gradients = m_gradients.cpu()
        return m_gradients

    @staticmethod
    # def serialize_model(model: torch.nn.Module) -> torch.Tensor:
    def serialize_model(rank_model: torch.nn.Module, rank: int, server_rank: int) -> torch.Tensor:
        """Unfold model parameters
        
        Unfold every layer of model, concate all of tensors into one.
        Return a `torch.Tensor` with shape (size, ).

        Args:
            model (torch.nn.Module): model to serialize.
        """

        # all lora needs to be filled zero to server_rank
        diff_rank = server_rank - rank

        if (diff_rank == 0):
            parameters_list = [param.data.view(-1) for param in rank_model.parameters()]
        else:
            parameters_list = []
            hidden_size = 768
            padd_zero_length = diff_rank * hidden_size
            for name, parameter in rank_model.named_parameters():
                parameter_data_view = parameter.data.view(-1)
                if ("lora" in name):
                    parameter_data_view = torch.cat((parameter_data_view, torch.zeros(padd_zero_length, device=parameter_data_view.device)), 0)
                parameters_list.append(parameter_data_view)

        # print("rank8 len(parameters):", len(parameters))
        m_parameters = torch.cat(parameters_list)
        # martinc
        # print("rank8 m_parameters.size():", m_parameters.size())# torch.Size([125534212])
        # print("serialize_model rank:", str(rank), " m_parameters.size():", m_parameters.size())
        m_parameters = m_parameters.cpu()

        return m_parameters

    @staticmethod
    def deserialize_model(model: torch.nn.Module,
                          serialized_parameters: torch.Tensor, rank: int, server_rank: int,
                          mode="copy"):
        """Assigns serialized parameters to model.parameters.
        This is done by iterating through ``model.parameters()`` and assigning the relevant params in ``grad_update``.
        NOTE: this function manipulates ``model.parameters``.

        Args:
            model (torch.nn.Module): model to deserialize.
            serialized_parameters (torch.Tensor): serialized model parameters.
            mode (str): deserialize mode. "copy" or "add".
        """

        # print("deserialize_model rank:", rank)
        current_index = 0  # keep track of where to read from grad_update
        for name, parameter in model.named_parameters():
            numel = parameter.data.numel()
            size = parameter.data.size()

            if ("lora" in name):
                diff_rank = server_rank - rank
                hidden_size = 768
                padd_zero_length = diff_rank * hidden_size
                numel += padd_zero_length
            else:
                if mode == "copy":
                    parameter.data.copy_(
                        serialized_parameters[current_index:current_index +
                                            numel].view(size))
                elif mode == "add":
                    parameter.data.add_(
                        serialized_parameters[current_index:current_index +
                                            numel].view(size))
                else:
                    raise ValueError(
                        "Invalid deserialize mode {}, require \"copy\" or \"add\" "
                        .format(mode))

            current_index += numel
        
        # original
        """
        current_index = 0  # keep track of where to read from grad_update
        for parameter in model.parameters():
            numel = parameter.data.numel()
            size = parameter.data.size()
            if mode == "copy":
                parameter.data.copy_(
                    serialized_parameters[current_index:current_index +
                                          numel].view(size))
            elif mode == "add":
                parameter.data.add_(
                    serialized_parameters[current_index:current_index +
                                          numel].view(size))
            else:
                raise ValueError(
                    "Invalid deserialize mode {}, require \"copy\" or \"add\" "
                    .format(mode))
            current_index += numel
        """

    @staticmethod
    def original_serialize_model(model: torch.nn.Module) -> torch.Tensor:
        """Unfold model parameters
        
        Unfold every layer of model, concate all of tensors into one.
        Return a `torch.Tensor` with shape (size, ).

        Args:
            model (torch.nn.Module): model to serialize.
        """

        parameters = [param.data.view(-1) for param in model.parameters()]
        m_parameters = torch.cat(parameters)
        m_parameters = m_parameters.cpu()

        return m_parameters

    @staticmethod
    def original_deserialize_model(model: torch.nn.Module,
                          serialized_parameters: torch.Tensor,
                          mode="copy"):
        """Assigns serialized parameters to model.parameters.
        This is done by iterating through ``model.parameters()`` and assigning the relevant params in ``grad_update``.
        NOTE: this function manipulates ``model.parameters``.

        Args:
            model (torch.nn.Module): model to deserialize.
            serialized_parameters (torch.Tensor): serialized model parameters.
            mode (str): deserialize mode. "copy" or "add".
        """

        current_index = 0  # keep track of where to read from grad_update
        for parameter in model.parameters():
            numel = parameter.data.numel()
            size = parameter.data.size()
            if mode == "copy":
                parameter.data.copy_(
                    serialized_parameters[current_index:current_index +
                                          numel].view(size))
            elif mode == "add":
                parameter.data.add_(
                    serialized_parameters[current_index:current_index +
                                          numel].view(size))
            else:
                raise ValueError(
                    "Invalid deserialize mode {}, require \"copy\" or \"add\" "
                    .format(mode))
            current_index += numel