import math

import torch
from torch import nn
from transformers import BertModel, RobertaModel


class EMA(object):

    def __init__(self, model, mu, level='batch', n=1):
        # self.ema_model = copy.deepcopy(model)
        self.mu = mu
        self.level = level
        self.n = n
        self.cnt = self.n
        self.shadow = {}
        for name, param in model.named_parameters():
            if True or param.requires_grad:
                self.shadow[name] = param.data

    def _update(self, model):
        for name, param in model.named_parameters():
            if True or param.requires_grad:
                new_average = (1 - self.mu) * param.data + \
                    self.mu * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def set_weights(self, ema_model):
        for name, param in ema_model.named_parameters():
            if True or param.requires_grad:
                param.data = self.shadow[name]

    def on_batch_end(self, model):
        if self.level == 'batch':
            self.cnt -= 1
            if self.cnt == 0:
                self._update(model)
                self.cnt = self.n

    def on_epoch_end(self, model):
        if self.level == 'epoch':
            self._update(model)


class SoftArgmax1D(torch.nn.Module):
    """
    Implementation of a 1d soft arg-max function as an nn.Module, so that we can differentiate through arg-max operations.
    """

    def __init__(self, base_index=0, step_size=1, beta=5., device='cpu'):
        """
        The "arguments" are base_index, base_index+step_size, base_index+2*step_size, ... and so on for
        arguments at indices 0, 1, 2, ....
        Assumes that the input to this layer will be a batch of 1D tensors (so a 2D tensor).
        :param base_index: Remember a base index for 'indices' for the input
        :param step_size: Step size for 'indices' from the input
        """
        super(SoftArgmax1D, self).__init__()
        self.base_index = base_index
        self.step_size = step_size
        self.softmax = torch.nn.Softmax(dim=1).to(device)
        self.beta = beta
        self.device = device

    def forward(self, x):
        """
        Compute the forward pass of the 1D soft arg-max function as defined below:
        SoftArgMax(x) = \sum_i (i * softmax(x)_i)
        :param x: The input to the soft arg-max layer
        :return: Output of the soft arg-max layer
        """
        smax = self.softmax(x * self.beta)
        end_index = self.base_index + x.size()[1] * self.step_size
        indices = torch.arange(
            start=self.base_index,
            end=end_index,
            step=self.step_size).to(self.device).float()
        return torch.matmul(smax, indices)


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
            if isinstance(pretrained_model_name_or_path, str):
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
            if isinstance(pretrained_model_name_or_path, str):
                self.model = RobertaModel.from_pretrained(
                    pretrained_model_name_or_path)
            else:
                # for sub
                self.model = RobertaModel(pretrained_model_name_or_path)
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
                encoder_attention_mask=None, special_tokens_mask=None):
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

        # special tokes を -inf で mask
        if special_tokens_mask is not None:
            inf = torch.tensor(float('inf')).to(logits_head.device)
            logits_head = logits_head.where(special_tokens_mask == 0, -inf)
            # we use [head:tail] type indexing,
            # so tail mask should be shifted
            tail_special_tokens_mask = torch.cat(
                [special_tokens_mask[:, -1:],
                 special_tokens_mask[:, :-1]],
                dim=1)
            logits_tail = logits_tail.where(
                tail_special_tokens_mask == 0, -inf)

        # add hidden states and attention if they are here
        outputs = ((logits_head, logits_tail),)

        return outputs  # logits, (hidden_states), (attentions)

    def resize_token_embeddings(self, token_num):
        self.model.resize_token_embeddings(token_num)


class Conv1dSame(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, dilation=1):
        super().__init__()
        self.cut_last_element = (
            kernel_size %
            2 == 0 and stride == 1 and dilation %
            2 == 1)
        self.padding = math.ceil(
            (1 - stride + dilation * (kernel_size - 1)) / 2)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            stride=stride,
            dilation=dilation)

    def forward(self, x):
        if self.cut_last_element:
            return self.conv(x)[:, :, :-1]
        else:
            return self.conv(x)


class RobertaModelWDualMultiClassClassifierAndSegmentationHead(
        RobertaModelWDualMultiClassClassifierHead):
    def __init__(self, num_labels, pretrained_model_name_or_path):
        super().__init__(num_labels, pretrained_model_name_or_path)
        self.classifier_conv_segmentation = nn.Conv1d(
            self.model.pooler.dense.out_features, 1, 1)
        self.add_module(
            'conv_output_segmentation',
            self.classifier_conv_segmentation)

    def forward(self, input_ids=None, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, encoder_hidden_states=None,
                encoder_attention_mask=None, special_tokens_mask=None):
        outputs = self.model(input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             position_ids=position_ids,
                             head_mask=head_mask,
                             inputs_embeds=inputs_embeds,
                             encoder_hidden_states=encoder_hidden_states,
                             encoder_attention_mask=encoder_attention_mask)
        output = outputs[0]
        output = torch.transpose(output, 1, 2)
        output = self.dropout(output)
        logits_head = self.classifier_conv_head(output).squeeze()
        logits_tail = self.classifier_conv_tail(output).squeeze()
        logits_segmentation = self.classifier_conv_segmentation(
            output).squeeze()

        # special tokes を -inf で mask
        if special_tokens_mask is not None:
            inf = torch.tensor(float('inf')).to(logits_head.device)
            logits_head = logits_head.where(special_tokens_mask == 0, -inf)
            # we use [head:tail] type indexing,
            # so tail mask should be shifted
            tail_special_tokens_mask = torch.cat(
                [special_tokens_mask[:, -1:],
                 special_tokens_mask[:, :-1]],
                dim=1)
            logits_tail = logits_tail.where(
                tail_special_tokens_mask == 0, -inf)

        # add hidden states and attention if they are here
        outputs = ((logits_head, logits_tail,
                    logits_segmentation),) + outputs[2:]

        return outputs  # logits, (hidden_states), (attentions)


class RobertaModelHeadClassAndAnchorHead(nn.Module):
    def __init__(self, num_labels, pretrained_model_name_or_path):
        super().__init__()
        if pretrained_model_name_or_path:
            if isinstance(pretrained_model_name_or_path, str):
                self.model = RobertaModel.from_pretrained(
                    pretrained_model_name_or_path)
            else:
                # for sub
                self.model = RobertaModel(pretrained_model_name_or_path)
        else:
            raise NotImplementedError
        self.num_labels = num_labels
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.classifier_conv_head = nn.Conv1d(
            self.model.pooler.dense.out_features, 1, 1)
        self.classifier_conv_tail = nn.Conv1d(
            self.model.pooler.dense.out_features, 1, 1)
        self.classifier_anchor = nn.Linear(num_labels, 1)

    def forward(self, input_ids=None, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, encoder_hidden_states=None,
                encoder_attention_mask=None, special_tokens_mask=None):
        outputs = self.model(input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             position_ids=position_ids,
                             head_mask=head_mask,
                             inputs_embeds=inputs_embeds,
                             encoder_hidden_states=encoder_hidden_states,
                             encoder_attention_mask=encoder_attention_mask)

        output = outputs[0]
        output = torch.transpose(output, 1, 2)
        output = self.dropout(output)
        logits_head = self.classifier_conv_head(output).squeeze()
        logits_tail = self.classifier_conv_tail(output).squeeze()
        anchor_value = self.classifier_anchor(logits_tail)
        anchor_value = self.relu(anchor_value)

        # special tokes を -inf で mask
        if special_tokens_mask is not None:
            raise NotImplementedError()

        # add hidden states and attention if they are here
        outputs = ((logits_head, anchor_value),)

        return outputs  # logits, (hidden_states), (attentions)

    def resize_token_embeddings(self, token_num):
        self.model.resize_token_embeddings(token_num)


class RobertaModelWDualMultiClassClassifierHeadV2(nn.Module):
    def __init__(self, num_labels, pretrained_model_name_or_path):
        super().__init__()
        if pretrained_model_name_or_path:
            if isinstance(pretrained_model_name_or_path, str):
                self.model = RobertaModel.from_pretrained(
                    pretrained_model_name_or_path)
            else:
                # for sub
                self.model = RobertaModel(pretrained_model_name_or_path)
        else:
            raise NotImplementedError
        self.num_labels = num_labels
        self.dropout = nn.Dropout(0.2)
        self.leaky_relu = nn.LeakyReLU()
        self.classifier_conv_head = Conv1dSame(
            self.model.pooler.dense.out_features, 128, 2)
        self.classifier_conv_tail = Conv1dSame(
            self.model.pooler.dense.out_features, 128, 2)
        self.classifier_conv_head_2 = Conv1dSame(128, 64, 2)
        self.classifier_conv_tail_2 = Conv1dSame(128, 64, 2)
        self.classifier_dense_head = nn.Linear(64, 1)
        self.classifier_dense_tail = nn.Linear(64, 1)
        self.add_module('conv_output_head', self.classifier_conv_head)
        self.add_module('conv_output_tail', self.classifier_conv_tail)
        self.add_module('conv_output_head_2', self.classifier_conv_head_2)
        self.add_module('conv_output_tail_2', self.classifier_conv_tail_2)

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

        output = outputs[0]
        output = torch.transpose(output, 1, 2)
        output = self.dropout(output)
        logits_head = self.classifier_conv_head(output)
        logits_head = self.leaky_relu(logits_head)
        logits_head = self.classifier_conv_head_2(logits_head)
        logits_head = torch.transpose(logits_head, 1, 2)
        logits_head = self.classifier_dense_head(logits_head).squeeze()
        logits_tail = self.classifier_conv_tail(output)
        logits_tail = self.leaky_relu(logits_tail)
        logits_tail = self.classifier_conv_tail_2(logits_tail)
        logits_tail = torch.transpose(logits_tail, 1, 2)
        logits_tail = self.classifier_dense_tail(logits_tail).squeeze()

        # add hidden states and attention if they are here
        outputs = ((logits_head, logits_tail),) + outputs[2:]

        return outputs  # logits, (hidden_states), (attentions)

    def resize_token_embeddings(self, token_num):
        self.model.resize_token_embeddings(token_num)


class RobertaModelWDualMultiClassClassifierHeadV3(nn.Module):
    def __init__(self, num_labels, pretrained_model_name_or_path):
        super().__init__()
        if pretrained_model_name_or_path:
            if isinstance(pretrained_model_name_or_path, str):
                self.model = RobertaModel.from_pretrained(
                    pretrained_model_name_or_path,
                    output_hidden_states=True)
            else:
                # for sub
                self.model = RobertaModel(
                    pretrained_model_name_or_path,
                    output_hidden_states=True)
        else:
            raise NotImplementedError
        self.num_labels = num_labels
        self.dropout = nn.Dropout(0.2)
        self.classifier_head_tail = nn.Linear(
            self.model.pooler.dense.out_features * 2, 2)

    def forward(self, input_ids=None, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, encoder_hidden_states=None,
                encoder_attention_mask=None, special_tokens_mask=None):
        outputs = self.model(input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             position_ids=position_ids,
                             head_mask=head_mask,
                             inputs_embeds=inputs_embeds,
                             encoder_hidden_states=encoder_hidden_states,
                             encoder_attention_mask=encoder_attention_mask)
        output = outputs[2]
        output = torch.cat((output[-1], output[-2]), dim=-1)
        output = self.dropout(output)
        logits = self.classifier_head_tail(output)
        logits_head, logits_tail = logits.split(1, dim=-1)

        # special tokes を -inf で mask
        if special_tokens_mask is not None:
            inf = torch.tensor(float('inf')).to(logits_head.device)
            logits_head = logits_head.where(special_tokens_mask == 0, -inf)
            # we use [head:tail] type indexing,
            # so tail mask should be shifted
            tail_special_tokens_mask = torch.cat(
                [special_tokens_mask[:, -1:],
                 special_tokens_mask[:, :-1]],
                dim=1)
            logits_tail = logits_tail.where(
                tail_special_tokens_mask == 0, -inf)

        logits_head = logits_head.squeeze(-1)
        logits_tail = logits_tail.squeeze(-1)

        # add hidden states and attention if they are here
        outputs = ((logits_head, logits_tail),)

        return outputs  # logits, (hidden_states), (attentions)

    def resize_token_embeddings(self, token_num):
        self.model.resize_token_embeddings(token_num)


class RobertaModelWDualMultiClassClassifierHeadV4(nn.Module):
    def __init__(self, num_labels, pretrained_model_name_or_path):
        super().__init__()
        if pretrained_model_name_or_path:
            if isinstance(pretrained_model_name_or_path, str):
                self.model = RobertaModel.from_pretrained(
                    pretrained_model_name_or_path,
                    output_hidden_states=True)
            else:
                # for sub
                self.model = RobertaModel(
                    pretrained_model_name_or_path,
                    output_hidden_states=True)
        else:
            raise NotImplementedError
        self.num_labels = num_labels
        self.dropout = nn.Dropout(0.2)
        self.leaky_relu = nn.LeakyReLU()
        # self.classifier_conv_adp = nn.Conv1d(
        #     self.model.pooler.dense.out_features, 13, 1)
        self.classifier_adp_weights = nn.Parameter(torch.randn(13))
        self.classifier_adp_weights.requires_grad = True
        self.classifier_conv_head = nn.Conv1d(
            self.model.pooler.dense.out_features, 1, 1)
        self.classifier_conv_tail = nn.Conv1d(
            self.model.pooler.dense.out_features, 1, 1)
        # self.add_module('conv_output_adp', self.classifier_conv_adp)
        self.add_module('conv_output_head', self.classifier_conv_head)
        self.add_module('conv_output_tail', self.classifier_conv_tail)

    def forward(self, input_ids=None, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, encoder_hidden_states=None,
                encoder_attention_mask=None, special_tokens_mask=None):
        outputs = self.model(input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             position_ids=position_ids,
                             head_mask=head_mask,
                             inputs_embeds=inputs_embeds,
                             encoder_hidden_states=encoder_hidden_states,
                             encoder_attention_mask=encoder_attention_mask)

        # adp_output = outputs[0]
        # adp_output = self.dropout(adp_output)
        # adp_output = torch.transpose(adp_output, 1, 2)
        # adp_output = self.classifier_conv_adp(adp_output).squeeze()
        # adp_output = torch.transpose(adp_output, 1, 2)
        # adp_output = adp_output.mean(dim=1)
        # adp_output /= adp_output.sum(dim=1).reshape(-1, 1)  # scale 1 sum

        output = outputs[2]
        w_sum = self.leaky_relu(self.classifier_adp_weights).sum()
        output = [output[i] * self.leaky_relu(self.classifier_adp_weights[i]) / w_sum
                  for i in range(13)]
        temp_output = output[0]
        for i in range(1, 13):
            temp_output += output[i]
        output = temp_output

        output = torch.transpose(output, 1, 2)
        output = self.dropout(output)
        logits_head = self.classifier_conv_head(output).squeeze()
        logits_tail = self.classifier_conv_tail(output).squeeze()

        # special tokes を -inf で mask
        if special_tokens_mask is not None:
            inf = torch.tensor(float('inf')).to(logits_head.device)
            logits_head = logits_head.where(special_tokens_mask == 0, -inf)
            # we use [head:tail] type indexing,
            # so tail mask should be shifted
            tail_special_tokens_mask = torch.cat(
                [special_tokens_mask[:, -1:],
                 special_tokens_mask[:, :-1]],
                dim=1)
            logits_tail = logits_tail.where(
                tail_special_tokens_mask == 0, -inf)

        # add hidden states and attention if they are here
        outputs = ((logits_head, logits_tail),)

        return outputs  # logits, (hidden_states), (attentions)

    def resize_token_embeddings(self, token_num):
        self.model.resize_token_embeddings(token_num)


class RobertaModelWDualMultiClassClassifierHeadV5(nn.Module):
    def __init__(self, num_labels, pretrained_model_name_or_path):
        super().__init__()
        if pretrained_model_name_or_path:
            if isinstance(pretrained_model_name_or_path, str):
                self.model = RobertaModel.from_pretrained(
                    pretrained_model_name_or_path,
                    output_hidden_states=True)
            else:
                # for sub
                self.model = RobertaModel(
                    pretrained_model_name_or_path,
                    output_hidden_states=True)
        else:
            raise NotImplementedError
        self.num_labels = num_labels
        self.dropout = nn.Dropout(0.2)
        self.classifier_conv_head = nn.Conv1d(
            self.model.pooler.dense.out_features * 2, 1, 1)
        self.classifier_conv_tail = nn.Conv1d(
            self.model.pooler.dense.out_features * 2, 1, 1)
        self.add_module('conv_output_head', self.classifier_conv_head)
        self.add_module('conv_output_tail', self.classifier_conv_tail)

    def forward(self, input_ids=None, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, encoder_hidden_states=None,
                encoder_attention_mask=None, special_tokens_mask=None):
        outputs = self.model(input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             position_ids=position_ids,
                             head_mask=head_mask,
                             inputs_embeds=inputs_embeds,
                             encoder_hidden_states=encoder_hidden_states,
                             encoder_attention_mask=encoder_attention_mask)

        output = torch.cat([outputs[2][-1], outputs[2][-2]], dim=-1)
        output = torch.transpose(output, 1, 2)
        output = self.dropout(output)
        logits_head = self.classifier_conv_head(output).squeeze()
        logits_tail = self.classifier_conv_tail(output).squeeze()

        # special tokes を -inf で mask
        if special_tokens_mask is not None:
            inf = torch.tensor(float('inf')).to(logits_head.device)
            logits_head = logits_head.where(special_tokens_mask == 0, -inf)
            # we use [head:tail] type indexing,
            # so tail mask should be shifted
            tail_special_tokens_mask = torch.cat(
                [special_tokens_mask[:, -1:],
                 special_tokens_mask[:, :-1]],
                dim=1)
            logits_tail = logits_tail.where(
                tail_special_tokens_mask == 0, -inf)

        # add hidden states and attention if they are here
        outputs = ((logits_head, logits_tail),)

        return outputs  # logits, (hidden_states), (attentions)

    def resize_token_embeddings(self, token_num):
        self.model.resize_token_embeddings(token_num)


class RobertaModelWDualMultiClassClassifierHeadV6(
        RobertaModelWDualMultiClassClassifierHead):
    def __init__(self, num_labels, pretrained_model_name_or_path):
        super().__init__(num_labels, pretrained_model_name_or_path)
        self.classifier_regen_tail = nn.Linear(num_labels * 2, num_labels)

    def forward(self, input_ids=None, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, encoder_hidden_states=None,
                encoder_attention_mask=None, special_tokens_mask=None):

        outputs = super().forward(input_ids, attention_mask,
                                  token_type_ids, position_ids, head_mask,
                                  inputs_embeds, encoder_hidden_states,
                                  encoder_attention_mask, special_tokens_mask)

        # tail を head に依存させる
        logits_head_tail = torch.cat(outputs[0], dim=-1)
        logits_tail = self.classifier_regen_tail(logits_head_tail)
        outputs[0][1] = logits_tail

        return outputs  # logits, (hidden_states), (attentions)


class RobertaModelWDualMultiClassClassifierHeadV7(nn.Module):
    def __init__(self, num_labels, pretrained_model_name_or_path):
        super().__init__()
        if pretrained_model_name_or_path:
            if isinstance(pretrained_model_name_or_path, str):
                self.model = RobertaModel.from_pretrained(
                    pretrained_model_name_or_path)
            else:
                # for sub
                self.model = RobertaModel(pretrained_model_name_or_path)
        else:
            raise NotImplementedError
        self.num_labels = num_labels
        self.dropout = nn.Dropout(0.2)
        self.classifier_conv_head = nn.Conv1d(
            self.model.pooler.dense.out_features, 1, 2, padding=1)
        self.classifier_conv_tail = nn.Conv1d(
            self.model.pooler.dense.out_features, 1, 2, padding=1)
        self.add_module('conv_output_head', self.classifier_conv_head)
        self.add_module('conv_output_tail', self.classifier_conv_tail)

    def forward(self, input_ids=None, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, encoder_hidden_states=None,
                encoder_attention_mask=None, special_tokens_mask=None):
        outputs = self.model(input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             position_ids=position_ids,
                             head_mask=head_mask,
                             inputs_embeds=inputs_embeds,
                             encoder_hidden_states=encoder_hidden_states,
                             encoder_attention_mask=encoder_attention_mask)

        output = outputs[0]
        output = torch.transpose(output, 1, 2)
        output = self.dropout(output)
        logits_head = self.classifier_conv_head(output).squeeze()[:, :-1]
        logits_tail = self.classifier_conv_tail(output.flip(dims=(2, ))).squeeze()[:, 1:].flip(dims=(1, ))

        # special tokes を -inf で mask
        if special_tokens_mask is not None:
            inf = torch.tensor(float('inf')).to(logits_head.device)
            logits_head = logits_head.where(special_tokens_mask == 0, -inf)
            # we use [head:tail] type indexing,
            # so tail mask should be shifted
            tail_special_tokens_mask = torch.cat(
                [special_tokens_mask[:, -1:],
                 special_tokens_mask[:, :-1]],
                dim=1)
            logits_tail = logits_tail.where(
                tail_special_tokens_mask == 0, -inf)

        # add hidden states and attention if they are here
        outputs = ((logits_head, logits_tail),)

        return outputs  # logits, (hidden_states), (attentions)

    def resize_token_embeddings(self, token_num):
        self.model.resize_token_embeddings(token_num)


class RobertaModelWDualMultiClassClassifierAndSegmentationHeadV4(
        RobertaModelWDualMultiClassClassifierHeadV4):
    def __init__(self, num_labels, pretrained_model_name_or_path):
        super().__init__(num_labels, pretrained_model_name_or_path)
        self.classifier_conv_segmentation = nn.Conv1d(
            self.model.pooler.dense.out_features, 1, 1)
        self.add_module(
            'conv_output_segmentation',
            self.classifier_conv_segmentation)

    def forward(self, input_ids=None, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, encoder_hidden_states=None,
                encoder_attention_mask=None, special_tokens_mask=None):
        outputs = self.model(input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             position_ids=position_ids,
                             head_mask=head_mask,
                             inputs_embeds=inputs_embeds,
                             encoder_hidden_states=encoder_hidden_states,
                             encoder_attention_mask=encoder_attention_mask)
        output = outputs[2]
        w_sum = self.leaky_relu(self.classifier_adp_weights).sum()
        output = [output[i] * self.leaky_relu(self.classifier_adp_weights[i]) / w_sum
                  for i in range(13)]
        temp_output = output[0]
        for i in range(1, 13):
            temp_output += output[i]
        output = temp_output

        output = torch.transpose(output, 1, 2)
        output = self.dropout(output)
        logits_head = self.classifier_conv_head(output).squeeze()
        logits_tail = self.classifier_conv_tail(output).squeeze()
        logits_segmentation = self.classifier_conv_segmentation(
            output).squeeze()

        # special tokes を -inf で mask
        if special_tokens_mask is not None:
            inf = torch.tensor(float('inf')).to(logits_head.device)
            logits_head = logits_head.where(special_tokens_mask == 0, -inf)
            # we use [head:tail] type indexing,
            # so tail mask should be shifted
            tail_special_tokens_mask = torch.cat(
                [special_tokens_mask[:, -1:],
                 special_tokens_mask[:, :-1]],
                dim=1)
            logits_tail = logits_tail.where(
                tail_special_tokens_mask == 0, -inf)

        # add hidden states and attention if they are here
        outputs = ((logits_head, logits_tail,
                    logits_segmentation),)

        return outputs  # logits, (hidden_states), (attentions)


class RobertaModelWDualMultiClassClassifierAndSegmentationHeadV5(
        RobertaModelWDualMultiClassClassifierHeadV5):
    def __init__(self, num_labels, pretrained_model_name_or_path):
        super().__init__(num_labels, pretrained_model_name_or_path)
        self.classifier_conv_segmentation = nn.Conv1d(
            self.model.pooler.dense.out_features * 2, 1, 1)
        self.add_module(
            'conv_output_segmentation',
            self.classifier_conv_segmentation)

    def forward(self, input_ids=None, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, encoder_hidden_states=None,
                encoder_attention_mask=None, special_tokens_mask=None):
        outputs = self.model(input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             position_ids=position_ids,
                             head_mask=head_mask,
                             inputs_embeds=inputs_embeds,
                             encoder_hidden_states=encoder_hidden_states,
                             encoder_attention_mask=encoder_attention_mask)

        output = torch.cat([outputs[2][-1], outputs[2][-2]], dim=-1)

        output = torch.transpose(output, 1, 2)
        output = self.dropout(output)
        logits_head = self.classifier_conv_head(output).squeeze()
        logits_tail = self.classifier_conv_tail(output).squeeze()
        logits_segmentation = self.classifier_conv_segmentation(
            output).squeeze()

        # special tokes を -inf で mask
        if special_tokens_mask is not None:
            inf = torch.tensor(float('inf')).to(logits_head.device)
            logits_head = logits_head.where(special_tokens_mask == 0, -inf)
            # we use [head:tail] type indexing,
            # so tail mask should be shifted
            tail_special_tokens_mask = torch.cat(
                [special_tokens_mask[:, -1:],
                 special_tokens_mask[:, :-1]],
                dim=1)
            logits_tail = logits_tail.where(
                tail_special_tokens_mask == 0, -inf)

        # add hidden states and attention if they are here
        outputs = ((logits_head, logits_tail,
                    logits_segmentation),)

        return outputs  # logits, (hidden_states), (attentions)


class RobertaModelWDualMultiClassClassifierAndSegmentationHeadV6(
        RobertaModelWDualMultiClassClassifierAndSegmentationHead):
    def __init__(self, num_labels, pretrained_model_name_or_path):
        super().__init__(num_labels, pretrained_model_name_or_path)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.classifier_regen_tail = nn.Linear(num_labels * 2, num_labels)

    def forward(self, input_ids=None, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, encoder_hidden_states=None,
                encoder_attention_mask=None, special_tokens_mask=None):

        outputs = super().forward(input_ids, attention_mask,
                                  token_type_ids, position_ids, head_mask,
                                  inputs_embeds, encoder_hidden_states,
                                  encoder_attention_mask, special_tokens_mask)

        # tail を head に依存させる
        logits_head_tail = torch.cat(outputs[0][0:2], dim=-1)
        logits_head_tail = self.relu(logits_head_tail)
        logits_head_tail = self.dropout2(logits_head_tail)
        logits_tail = self.classifier_regen_tail(logits_head_tail)
        outputs = ((outputs[0][0], logits_tail, outputs[0][2]), )

        return outputs  # logits, (hidden_states), (attentions)


class RobertaModelWDualMultiClassClassifierAndSegmentationHeadV7(
        RobertaModelWDualMultiClassClassifierHeadV7):
    def __init__(self, num_labels, pretrained_model_name_or_path):
        super().__init__(num_labels, pretrained_model_name_or_path)
        self.classifier_conv_segmentation = nn.Conv1d(
            self.model.pooler.dense.out_features, 1, 3, padding=1)
        self.add_module(
            'conv_output_segmentation',
            self.classifier_conv_segmentation)

    def forward(self, input_ids=None, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, encoder_hidden_states=None,
                encoder_attention_mask=None, special_tokens_mask=None):
        outputs = self.model(input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             position_ids=position_ids,
                             head_mask=head_mask,
                             inputs_embeds=inputs_embeds,
                             encoder_hidden_states=encoder_hidden_states,
                             encoder_attention_mask=encoder_attention_mask)
        output = outputs[0]
        output = torch.transpose(output, 1, 2)
        output = self.dropout(output)
        logits_head = self.classifier_conv_head(output).squeeze()
        logits_tail = self.classifier_conv_tail(output).squeeze()
        logits_segmentation = self.classifier_conv_segmentation(
            output).squeeze()

        # special tokes を -inf で mask
        if special_tokens_mask is not None:
            inf = torch.tensor(float('inf')).to(logits_head.device)
            logits_head = logits_head.where(special_tokens_mask == 0, -inf)
            # we use [head:tail] type indexing,
            # so tail mask should be shifted
            tail_special_tokens_mask = torch.cat(
                [special_tokens_mask[:, -1:],
                 special_tokens_mask[:, :-1]],
                dim=1)
            logits_tail = logits_tail.where(
                tail_special_tokens_mask == 0, -inf)

        # add hidden states and attention if they are here
        outputs = ((logits_head, logits_tail,
                    logits_segmentation),) + outputs[2:]

        return outputs  # logits, (hidden_states), (attentions)


class RobertaModelWDualMultiClassClassifierAndCumsumSegmentationHead(
        RobertaModelWDualMultiClassClassifierHead):
    def __init__(self, num_labels, pretrained_model_name_or_path):
        super().__init__(num_labels, pretrained_model_name_or_path)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

    def forward(self, input_ids=None, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, encoder_hidden_states=None,
                encoder_attention_mask=None, special_tokens_mask=None):
        outputs = self.model(input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             position_ids=position_ids,
                             head_mask=head_mask,
                             inputs_embeds=inputs_embeds,
                             encoder_hidden_states=encoder_hidden_states,
                             encoder_attention_mask=encoder_attention_mask)
        output = outputs[0]
        output = torch.transpose(output, 1, 2)
        output = self.dropout(output)
        logits_head = self.classifier_conv_head(output).squeeze()
        logits_tail = self.classifier_conv_tail(output).squeeze()

        prob_head = self.softmax(logits_head.double())
        prob_tail = self.softmax(logits_tail.double())
        cumsum_head = torch.cumsum(prob_head, dim=1)
        cumsum_tail = torch.cumsum(prob_tail, dim=1)
        rev_prob_head = self.softmax(logits_head.double()).flip(dims=(1, ))
        rev_prob_tail = self.softmax(logits_tail.double()).flip(dims=(1, ))
        rev_cumsum_head = torch.cumsum(rev_prob_head, dim=1).flip(dims=(1, ))
        rev_cumsum_tail = torch.cumsum(rev_prob_tail, dim=1).flip(dims=(1, ))
        cumsum_pred = self.relu(cumsum_head - cumsum_tail) + 1e-7
        cumsum_pred += self.relu(rev_cumsum_tail - rev_cumsum_head) + 1e-7
        cumsum_pred /= 2

        # special tokes を -inf で mask
        if special_tokens_mask is not None:
            inf = torch.tensor(float('inf')).to(logits_head.device)
            logits_head = logits_head.where(special_tokens_mask == 0, -inf)
            # we use [head:tail] type indexing,
            # so tail mask should be shifted
            tail_special_tokens_mask = torch.cat(
                [special_tokens_mask[:, -1:],
                 special_tokens_mask[:, :-1]],
                dim=1)
            logits_tail = logits_tail.where(
                tail_special_tokens_mask == 0, -inf)

        # add hidden states and attention if they are here
        outputs = ((logits_head, logits_tail,
                    cumsum_pred),)

        return outputs  # logits, (hidden_states), (attentions)
