"""BaseModel for FedETuning"""

import copy
from abc import ABC
from utils import registry
from models.utils import PromptType

import torch
import torch.nn as nn
from transformers import trainer, AutoConfig
"""
from opendelta import AutoDeltaConfig
from opendelta.auto_delta import AutoDeltaModel
"""
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from peft import LoraModel, LoraConfig

import pickle


class BaseModels(nn.Module, ABC):
    def __init__(self, task_name, lora_r, lora_alpha):
        super().__init__()

        self.task_name = task_name

        config = registry.get("config")
        self.model_config = config.model_config
        self.rank = config.federated_config.rank
        self.logger = registry.get("logger")
        
        #martinc
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha

    def _build_config(self, **kwargs):
        auto_config = AutoConfig.from_pretrained(
            self.model_config.config_name if self.model_config.config_name else self.model_config.model_name_or_path,
            finetuning_task=self.task_name if self.task_name else None,
            # cache_dir=self.model_config.cache_dir,
            revision=self.model_config.model_revision,
            use_auth_token=True if self.model_config.use_auth_token else None,
            **kwargs
        )
        return auto_config

    def _build_model(self):
        backbone = self._add_base_model()

        if getattr(self.model_config, "permutation_layers", None):
            backbone = self._add_permutate_layers(backbone)

        if self.model_config.tuning_type:
            backbone = self._add_delta_model(backbone)

        return backbone

    def _add_base_model(self):
        raise NotImplementedError

    def _add_permutate_layers(self, backbone):
        # TODO only support BERT-NLU Task
        bert_modules = self.get_bert_module(backbone)
        old_modules = bert_modules.encoder.layer
        scrambled_modules = torch.nn.ModuleList()
        # Now iterate over all layers,
        # appending to the new module list according to the new order.
        if self.rank > 0:
            permutation = self.model_config.client_model_layers
        else:
            permutation = self.model_config.server_model_layers
        self.logger.debug(f"model's layer: {permutation}")
        for i in permutation:
            assert i <= 11, permutation
            scrambled_modules.append(old_modules[i])

        # Create a copy of the model, modify it with the new list, and return
        backbone_copy = copy.deepcopy(backbone)
        bert_modules_copy = self.get_bert_module(backbone_copy)
        bert_modules_copy.encoder.layer = scrambled_modules
        return backbone_copy

    def _add_delta_model(self, backbone):

        if any([True for PType in PromptType if PType in self.model_config.tuning_type]):
            # prefix tuning maybe in OpenDelta
            ...
        else:
            delta_args = registry.get("delta_config")

            peft_config = LoraConfig(
                # task_type=TaskType.SEQ_CLS, inference_mode=False, r=delta_args["lora_r"], lora_alpha=delta_args["lora_alpha"], lora_dropout=0.0, target_modules=["query", "value"]# "attn.q", "attn.v"
                task_type=TaskType.SEQ_CLS, inference_mode=False, r=self.lora_r, lora_alpha=self.lora_alpha, lora_dropout=0.0, target_modules=["query", "value"]# "attn.q", "attn.v"
            )

            backbone = get_peft_model(backbone, peft_config)
            """
            with open("opendelta_entiremodel_lora_roberta_port_10020_r" + str(self.rank) + ".pkl", 'rb') as f:
                load_opendelta_dict = pickle.load(f)

            
            # delta_config = AutoDeltaConfig.from_dict(delta_args)
            # delta_model = AutoDeltaModel.from_config(delta_config, backbone_model=backbone)
            # delta_model.freeze_module(set_state_dict=True)
            
            # delta_model.log(delta_ratio=True, trainable_ratio=True, visualization=True)
            # self.logger.debug(delta_config)
            # self.logger.debug(backbone)
            # self.logger.debug(delta_args)

            load_opendelta_value = list(load_opendelta_dict.values())
            load_opendelta_key = list(load_opendelta_dict.keys())


            index = 0
            for key, backbone_layer in backbone.named_parameters():
                # print("index:", index, " key:", key)
                
                if(index == 249):
                    index = 245
                backbone_layer.data.copy_(load_opendelta_value[index])
                
                index += 1
            """
            """
            self.round_count = nn.ParameterDict({})
            round_value = torch.ones((1), dtype=self.weight.dtype, device=self.weight.device)
            self.round_count.update(nn.ParameterDict({"round_count_martinc": nn.Parameter(round_value)}))
            """

            """
            print("before backbone")
            print(backbone)

            backbone = nn.Sequential(
                backbone,
                torch.ones(1)
            )

            print("after backbone")
            print(backbone)
            """          

        return backbone

    def forward(self, inputs):
        raise NotImplementedError

    def get_bert_module(self, backbone):

        if self.model_config.model_type == "bert":
            return backbone.bert
        elif self.model_config.model_type == "roberta":
            return backbone.roberta
        elif self.model_config.model_type == "distilbert":
            return backbone.distilbert
        else:
            raise NotImplementedError
