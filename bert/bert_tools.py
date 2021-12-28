import torch
from transformers import BertModel, BertTokenizer

BERT_MODEL = None
BERT_TOK = None
weights_name = "bert-base-uncased"

OBJ_REF_TOK = "<obj_ref>"


def _load_bert_if_needed():
    global BERT_MODEL
    if BERT_MODEL is None:
        BERT_MODEL = BertModel.from_pretrained(weights_name)


def _load_tok_if_needed():
    global BERT_TOK
    if BERT_TOK is None:
        BERT_TOK = BertTokenizer.from_pretrained(weights_name)
        BERT_TOK.add_special_tokens({"additional_special_tokens": [OBJ_REF_TOK]})
        print("ding")


def bert_tokenize_instruction(instruction_str, return_special_token_indices=False):
    _load_tok_if_needed()
    tokenized_text = BERT_TOK.encode(instruction_str, add_special_tokens=True)
    if return_special_token_indices:
        obj_ref_tok_id = BERT_TOK.encode(OBJ_REF_TOK)[0]
        special_token_indices = [i for i, x in enumerate(tokenized_text) if x == obj_ref_tok_id]
        return tokenized_text, special_token_indices
    return tokenized_text


def bert_untokenize_instruction(tokenized_text):
    _load_tok_if_needed()
    raw_text = BERT_TOK.decode(tokenized_text)
    return raw_text


def bert_embed_string(nl_str):
    _load_bert_if_needed()
    tok_str = bert_tokenize_instruction(nl_str)
    # Encode text
    input_ids = torch.tensor([tok_str])
    with torch.no_grad():
        last_hidden_states = BERT_MODEL(input_ids)[0]  # Models outputs are now tuples
    return last_hidden_states


def bert_embed_string_batch(list_of_nl_strings):
    _load_bert_if_needed()
    tok_strings = [bert_tokenize_instruction(i) for i in list_of_nl_strings]
    raise NotImplementedError()