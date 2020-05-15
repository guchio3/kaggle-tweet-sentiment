import torch
from torch import nn
from transformers import BertModel, RobertaModel


class BertModelWBinaryMultiLabelClassifierHead(nn.Module):
    def __init__(self, num_labels, pretrained_model_name_or_path):
        super().__init__()
        if pretrained_model_name_or_path:
            self.model = BertModel.from_pretrained(
                pretrained_model_name_or_path)
        else:
            raise NotImplementedError
        self.num_labels = num_labels
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(
            self.model.pooler.dense.out_features, num_labels)
        self.add_module('fc_output', self.classifier)

    def forward(self, input_ids=None, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, encoder_hidden_states=None,
                encoder_attention_mask=None):
        outputs = self.model(input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             position_ids=position_ids,
                             head_mask=head_mask,
                             inputs_embeds=inputs_embeds,
                             encoder_hidden_states=encoder_hidden_states,
                             encoder_attention_mask=encoder_attention_mask)
        # pooled_output = outputs[1]
        pooled_output = torch.mean(outputs[0], dim=1)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # add hidden states and attention if they are here
        outputs = (logits,) + outputs[2:]

        # if labels is not None:
        #     loss = self.fobj(logits, labels)
        #     outputs = (loss,) + outputs
        # else:
        #     outputs = (None,) + outputs

        return outputs  # logits, (hidden_states), (attentions)

    def resize_token_embeddings(self, token_num):
        self.model.resize_token_embeddings(token_num)


class BertModelWDualMultiClassClassifierHead(nn.Module):
    def __init__(self, num_labels, pretrained_model_name_or_path):
        super().__init__()
        if pretrained_model_name_or_path:
            if type(pretrained_model_name_or_path) == 'str':
                self.model = BertModel.from_pretrained(
                    pretrained_model_name_or_path)
            else:
                # for sub
                self.model = BertModel(pretrained_model_name_or_path)
        else:
            raise NotImplementedError
        self.num_labels = num_labels
        self.dropout = nn.Dropout(0.2)
        self.classifier_conv_head = nn.Conv1d(
            self.model.pooler.dense.out_features, 1, 1)
        self.classifier_conv_tail = nn.Conv1d(
            self.model.pooler.dense.out_features, 1, 1)
        self.add_module('conv_output_head', self.classifier_conv_head)
        self.add_module('conv_output_tail', self.classifier_conv_tail)

        # self.classifier_head = nn.Linear(
        #     self.model.pooler.dense.out_features, num_labels)
        # self.classifier_tail = nn.Linear(
        #     self.model.pooler.dense.out_features, num_labels)

        # self.add_module('fc_output_head', self.classifier_head)
        # self.add_module('fc_output_tail', self.classifier_tail)

    def forward(self, input_ids=None, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, encoder_hidden_states=None,
                encoder_attention_mask=None):
        outputs = self.model(input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             position_ids=position_ids,
                             head_mask=head_mask,
                             inputs_embeds=inputs_embeds,
                             encoder_hidden_states=encoder_hidden_states,
                             encoder_attention_mask=encoder_attention_mask)
        # pooled_output = outputs[1]
        # pooled_output = torch.mean(outputs[0], dim=1)

        # pooled_output = self.dropout(pooled_output)
        # logits_head = self.classifier_head(pooled_output)
        # logits_tail = self.classifier_tail(pooled_output)
        output = outputs[0]
        output = torch.transpose(output, 1, 2)
        output = self.dropout(output)
        logits_head = self.classifier_conv_head(output).squeeze()
        logits_tail = self.classifier_conv_tail(output).squeeze()

        # add hidden states and attention if they are here
        outputs = ((logits_head, logits_tail),) + outputs[2:]

        return outputs  # logits, (hidden_states), (attentions)

    def resize_token_embeddings(self, token_num):
        self.model.resize_token_embeddings(token_num)


class RobertaModelWDualMultiClassClassifierHead(nn.Module):
    def __init__(self, num_labels, pretrained_model_name_or_path):
        super().__init__()
        if pretrained_model_name_or_path:
            self.model = RobertaModel.from_pretrained(
                pretrained_model_name_or_path)
        else:
            raise NotImplementedError
        self.num_labels = num_labels
        self.dropout = nn.Dropout(0.2)
        self.classifier_conv_head = nn.Conv1d(
            self.model.pooler.dense.out_features, 1, 1)
        self.classifier_conv_tail = nn.Conv1d(
            self.model.pooler.dense.out_features, 1, 1)
        self.add_module('conv_output_head', self.classifier_conv_head)
        self.add_module('conv_output_tail', self.classifier_conv_tail)

        # self.classifier_head = nn.Linear(
        #     self.model.pooler.dense.out_features, num_labels)
        # self.classifier_tail = nn.Linear(
        #     self.model.pooler.dense.out_features, num_labels)

        # self.add_module('fc_output_head', self.classifier_head)
        # self.add_module('fc_output_tail', self.classifier_tail)

    def forward(self, input_ids=None, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, encoder_hidden_states=None,
                encoder_attention_mask=None):
        outputs = self.model(input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             position_ids=position_ids,
                             head_mask=head_mask,
                             inputs_embeds=inputs_embeds,
                             encoder_hidden_states=encoder_hidden_states,
                             encoder_attention_mask=encoder_attention_mask)
        # pooled_output = outputs[1]
        # pooled_output = torch.mean(outputs[0], dim=1)

        # pooled_output = self.dropout(pooled_output)
        # logits_head = self.classifier_head(pooled_output)
        # logits_tail = self.classifier_tail(pooled_output)
        output = outputs[0]
        output = torch.transpose(output, 1, 2)
        output = self.dropout(output)
        logits_head = self.classifier_conv_head(output).squeeze()
        logits_tail = self.classifier_conv_tail(output).squeeze()

        # add hidden states and attention if they are here
        outputs = ((logits_head, logits_tail),) + outputs[2:]

        return outputs  # logits, (hidden_states), (attentions)

    def resize_token_embeddings(self, token_num):
        self.model.resize_token_embeddings(token_num)
