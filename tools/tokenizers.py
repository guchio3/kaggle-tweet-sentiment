import torch
from tokenizers import ByteLevelBPETokenizer


class myBertByteLevelBPETokenizer(ByteLevelBPETokenizer):
    def __len__(self, ):
        return self.get_vocab_size()

    def encode_plus(self, text, text_pair, add_special_tokens,
                    max_length, pad_to_max_length=True, return_tensor='pt',
                    return_token_type_ids=False, return_attention_mask=True):
        enc = self.encode(text)
        if add_special_tokens:
            input_ids = [101] + enc.ids + [102]
            if text_pair:
                pair_enc = self.encode(text_pair)
                input_ids += pair_enc.ids + [102]
            valid_input_len = len(input_ids)
            if pad_to_max_length:
                input_ids += [0] * (max_length - valid_input_len)

            res = {'input_ids': input_ids}

            if return_attention_mask:
                attention_mask = [1] * valid_input_len
                if pad_to_max_length:
                    attention_mask += [0] * (max_length - valid_input_len)
                res['attention_mask'] = attention_mask
        else:
            input_ids = enc.ids
            valid_input_len = len(input_ids)
            if pad_to_max_length:
                input_ids += [0] * (max_length - valid_input_len)

            res = {'input_ids': input_ids}
            if return_attention_mask:
                attention_mask = [1] * valid_input_len
                if pad_to_max_length:
                    attention_mask += [0] * (max_length - valid_input_len)
                res['attention_mask'] = attention_mask

        return res

    def decode(self, input_ids):
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.long().tolist()
        return super().decode(input_ids)


class myRobertaByteLevelBPETokenizer(ByteLevelBPETokenizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.special_tokens_map = {'pad_token': '<pad>'}

    def __len__(self, ):
        return self.get_vocab_size()

    def encode_plus(self, text, text_pair, add_special_tokens,
                    max_length, pad_to_max_length=True, return_tensor='pt',
                    return_token_type_ids=False, return_attention_mask=True):
        enc = self.encode(text)
        if add_special_tokens:
            input_ids = [0] + enc.ids + [2]
            if text_pair:
                pair_enc = self.encode(text_pair)
                input_ids += [2] + pair_enc.ids + [2]
            valid_input_len = len(input_ids)
            if pad_to_max_length:
                input_ids += [0] * (max_length - valid_input_len)
            # if return_tensor == 'pt':
            #     input_ids = torch.Tensor([input_ids])

            res = {'input_ids': input_ids}

            if return_attention_mask:
                attention_mask = [1] * valid_input_len
                if pad_to_max_length:
                    attention_mask += [0] * (max_length - valid_input_len)
                # if return_tensor == 'pt':
                #     attention_mask = torch.Tensor([attention_mask])
                res['attention_mask'] = attention_mask
        else:
            input_ids = enc.ids
            valid_input_len = len(input_ids)
            if pad_to_max_length:
                input_ids += [0] * (max_length - valid_input_len)
            # if return_tensor == 'pt':
            #     input_ids = torch.Tensor([input_ids])

            res = {'input_ids': input_ids}
            if return_attention_mask:
                attention_mask = [1] * valid_input_len
                if pad_to_max_length:
                    attention_mask += [0] * (max_length - valid_input_len)
                # if return_tensor == 'pt':
                #     attention_mask = torch.Tensor([attention_mask])
                res['attention_mask'] = attention_mask

        return res

    def decode(self, input_ids):
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.long().tolist()
        return super().decode(input_ids)
