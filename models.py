import numpy as np
import torch
import torch.nn.functional as F

from collections import defaultdict
from torch import nn
from torch.nn import CrossEntropyLoss, NLLLoss 
from torch.nn import Dropout
from transformers import BertConfig, BertModel, BertForMaskedLM
from typing import Any

class ActionBertModel(torch.nn.Module):
    def __init__(self,
                 model_name_or_path,
                 dropout,
                 num_action_labels):
        super(ActionBertModel, self).__init__()
        self.bert_model = BertModel.from_pretrained(model_name_or_path)

        self.dropout = Dropout(dropout)
        self.num_action_labels = num_action_labels
        self.action_classifier = nn.Linear(self.bert_model.config.hidden_size, num_action_labels)

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                action_label=None):
        pooled_output = self.bert_model(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        token_type_ids=token_type_ids)[1]
        action_logits = self.action_classifier(self.dropout(pooled_output))

        # Compute losses if labels provided
        if action_label is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(action_logits.view(-1, self.num_action_labels), action_label.type(torch.long))
        else:
            loss = torch.tensor(0)

        return action_logits, loss

class SchemaActionBertModel(torch.nn.Module):
    def __init__(self,
                 model_name_or_path,
                 dropout,
                 num_action_labels):
        super(SchemaActionBertModel, self).__init__()
        self.bert_model = BertModel.from_pretrained(model_name_or_path)

        self.dropout = Dropout(dropout)
        self.num_action_labels = num_action_labels
        self.action_classifier = nn.Linear(self.bert_model.config.hidden_size, num_action_labels)
        self.p_schema = nn.Linear(self.bert_model.config.hidden_size, 1)

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                tasks,
                action_label,
                sc_input_ids,
                sc_attention_mask,
                sc_token_type_ids,
                sc_tasks,
                sc_action_label,
                sc_full_graph):
        all_output, pooled_output = self.bert_model(input_ids=input_ids,
                                                    attention_mask=attention_mask,
                                                    token_type_ids=token_type_ids)
        action_logits = self.action_classifier(self.dropout(pooled_output))

        sc_all_output, sc_pooled_output = self.bert_model(input_ids=sc_input_ids,
                                                          attention_mask=sc_attention_mask,
                                                          token_type_ids=sc_token_type_ids)



        all_output_flat = all_output.view(-1, all_output.size(-1))
        i_probs = F.softmax(all_output_flat.mm(sc_all_output.view(-1, 768).t()), dim=-1).view(all_output_flat.size(0), -1, sc_input_ids.size(-1)).sum(dim=-1)

        probs = i_probs.view(input_ids.size(0), -1, i_probs.size(-1)).mean(dim=1)

        action_probs = torch.zeros(probs.size(0), self.num_action_labels).cuda().scatter_add(-1, sc_action_label.unsqueeze(0).repeat(probs.size(0), 1), probs)

        sc_prob = F.sigmoid(self.p_schema(pooled_output))

        action_lps = torch.log(action_probs+1e-10)

        # Compute losses if labels provided
        if action_label is not None:
            loss_fct = NLLLoss()
            loss = loss_fct(action_lps.view(-1, self.num_action_labels), action_label.type(torch.long))
        else:
            loss = torch.tensor(0)

        return action_lps, loss

    def predict(self,
                input_ids,
                attention_mask,
                token_type_ids,
                tasks,
                sc_all_output,
                sc_pooled_output,
                sc_tasks,
                sc_action_label,
                sc_full_graph):

        all_output, pooled_output = self.bert_model(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        token_type_ids=token_type_ids)
        action_logits = self.action_classifier(self.dropout(pooled_output))

        all_output_flat = all_output.view(-1, all_output.size(-1))
        i_probs = F.softmax(all_output_flat.mm(sc_all_output.view(-1, 768).t()), dim=-1).view(all_output_flat.size(0), -1, sc_all_output.size(-2)).sum(dim=-1)

        probs = i_probs.view(input_ids.size(0), -1, i_probs.size(-1)).mean(dim=1)

        # Zero out any attention across different tasks
        for i in range(probs.size(0)):
            for j in range(probs.size(1)):
                if tasks[i] != sc_tasks[j]:
                    probs[i,j] = 0


        action_probs = torch.zeros(probs.size(0), self.num_action_labels).cuda().scatter_add(-1, sc_action_label.unsqueeze(0).repeat(probs.size(0), 1), probs)

        sc_prob = F.sigmoid(self.p_schema(pooled_output))

        action_lps = torch.log(action_probs*sc_prob)

        return action_lps, 0
