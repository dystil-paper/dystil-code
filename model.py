import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
import utils
from utils.obs2text import gen_text_desc

ID_TO_ACTION = {0: 'left turn', 1: 'right turn', 2: 'move forward', 3: 'pick up', 4: 'drop', 5: 'toggle'}

action_tokenids_dict = {}
action_tokenids_dict['meta-llama/Meta-Llama-3.1-8B-Instruct'] = [2163, 1314, 3351, 3820, 6068, 15349]

# Define the DYSTIL Actor-Critic Agent Model

class ACModel(nn.Module):
    def __init__(self, core_model_name, header, strategies, num_actions, use_lora = False, lora_rank = None, freeze_lm = False, use_strategy = True, output_attentions = False, use_tensor = False):
        super().__init__()

        # Decide which components are enabled
        self.header = header
        self.use_strategy = use_strategy # whether the agent has access to a dynamically updated pool of strategies
        self.strategies = strategies
        self.output_attentions = output_attentions
        self.use_tensor = use_tensor
        self.use_lora = use_lora
        self.freeze_lm = freeze_lm
        self.num_actions = num_actions

        # Core Reasoning Language Model

        self.core_model_name = core_model_name

        if core_model_name in ['meta-llama/Meta-Llama-3.1-8B-Instruct']:
            self.tokenizer = AutoTokenizer.from_pretrained(core_model_name, token = "YOUR_TOKEN_HERE")

            self.text_model = AutoModelForCausalLM.from_pretrained(core_model_name, token = "YOUR_TOKEN_HERE")

            self.tokenizer.pad_token = '<pad>'
            self.tokenizer.pad_token_id = 128003

            self.text_model.config.pad_token_id = self.tokenizer.pad_token_id
        else:
            raise ValueError("Can't recognize the specified name of the core language model.")
        

        if self.use_lora:
            peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=lora_rank, lora_alpha=2*lora_rank, lora_dropout=0.1)

            self.text_model = get_peft_model(self.text_model, peft_config)
            self.text_model.print_trainable_parameters()
        
        self.action_ids = action_tokenids_dict[self.core_model_name][:self.num_actions]

        self.hidden_size = self.text_model.config.hidden_size

        # freeze all the parameters of the transformer language model except for the language modeling head (LM-head)
        if self.freeze_lm:
            for param in self.text_model.parameters():
                param.requires_grad = False
            for param in self.text_model.lm_head.parameters():
                param.requires_grad = True
    
        # Define critic's model (value head)
        self.critic = nn.Sequential(
            nn.Linear(self.hidden_size, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 1),
        )

        self.device = None

    def set_device(self):
        self.device = next(self.parameters()).device

    def forward(self, obs, h2obs, h2a, h1obs, h1a, ready_descriptions = None):

        if ready_descriptions == None:

            header = self.header

            if self.use_strategy:
                header += 'Rules to follow:\n' + self.strategies + '\n\n'

            descriptions = [header] * len(obs)

            for i in range(len(obs)):

                descriptions[i] += 'Goal of the agent: ' + obs[i].mission + '.\n'
                descriptions[i] += 'Observation: ' + (', '.join(gen_text_desc(obs[i].image.cpu().numpy())['descriptions'])).lower() + '.\n'
                descriptions[i] += 'Action:'
        else:
            descriptions = ready_descriptions

        if self.core_model_name in ['meta-llama/Meta-Llama-3.1-8B-Instruct']:
            inputs = self.tokenizer(descriptions, return_tensors='pt', padding=True, truncation = True, max_length = 2048).to(self.device)

            outputs = self.text_model(**inputs, output_attentions = self.output_attentions, output_hidden_states = True)

            positions = torch.eq(inputs['input_ids'], self.text_model.config.pad_token_id).int().argmax(-1) - 1

            logit_matrix = outputs.logits[torch.arange(len(descriptions)), positions]

            action_logits = logit_matrix[:, self.action_ids]

            embedding_vector = outputs.hidden_states[-1][torch.arange(len(descriptions)), positions]
        else:
            raise ValueError("Can't recognize the specified name of the core language model.")

        dist = Categorical(logits = action_logits)

        y = self.critic(embedding_vector)
        value = y.squeeze(1)

        if self.output_attentions:
            attention = outputs.cross_attentions[-1][:, :, -1, :]
            sum_attention = torch.sum(attention, dim = 1)
            normalized_attention = sum_attention / torch.sum(sum_attention, dim = 1, keepdim = True)

            return dist, value, normalized_attention
        else:
            return dist, value
        