import torch
import transformers
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
from typing import Tuple, Optional, Any, List

from .Model import Model

class MultiTokenEOSCriteria(transformers.StoppingCriteria):
    """Criteria to stop on the specified multi-token sequence."""

    def __init__(
        self,
        sequence: str,
        tokenizer: transformers.PreTrainedTokenizer,
        initial_decoder_input_length: int,
        batch_size: int,
    ) -> None:
        self.initial_decoder_input_length = initial_decoder_input_length
        self.done_tracker = [False] * batch_size
        self.sequence = sequence
        self.sequence_ids = tokenizer.encode(sequence, add_special_tokens=False)
        # print(sequence, self.sequence_ids)
        # we look back for 2 more tokens than it takes to encode our stop sequence
        # because tokenizers suck, and a model might generate `['\n', '\n']` but our `sequence` is `['\n\n']`
        # and we don't want to mistakenly not stop a generation because our
        # (string) stop sequence was output in a different tokenization

        # NOTE: there is a minor danger that this will end up looking back 2 tokens into the past, into the inputs to the model,
        # and stopping generation immediately as a result. With only 2 extra tokens of lookback, this risk is minimized
        # Additionally, in lookback_ids_batch we should prevent ever looking back into the inputs as described.
        self.sequence_id_len = len(self.sequence_ids) + 2
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # For efficiency, we compare the last n tokens where n is the number of tokens in the stop_sequence
        lookback_ids_batch = input_ids[:, self.initial_decoder_input_length :]

        lookback_ids_batch = lookback_ids_batch[:, -self.sequence_id_len :]

        lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)

        for i, done in enumerate(self.done_tracker):
            if not done:
                self.done_tracker[i] = self.sequence in lookback_tokens_batch[i]
        return False not in self.done_tracker

def stop_sequences_criteria(
    tokenizer: transformers.PreTrainedTokenizer,
    stop_sequences: List[str],
    initial_decoder_input_length: int,
    batch_size: int,
) -> transformers.StoppingCriteriaList:
    return transformers.StoppingCriteriaList(
        [
            *[
                MultiTokenEOSCriteria(
                    sequence, tokenizer, initial_decoder_input_length, batch_size
                )
                for sequence in stop_sequences
            ],
        ]
    )

class Llama(Model):
    def __init__(self, config):
        super().__init__(config)
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        self.device = config["params"]["device"]
        self.max_output_tokens = config["params"]["max_output_tokens"]

        api_pos = int(config["api_key_info"]["api_key_use"])
        hf_token = config["api_key_info"]["api_keys"][api_pos]

        #self.tokenizer = LlamaTokenizer.from_pretrained(self.name, use_auth_token=hf_token)
        #self.model = LlamaForCausalLM.from_pretrained(self.name, device_map="auto", torch_dtype=torch.float16, use_auth_token=hf_token)

        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(self.name, device_map="auto", torch_dtype=torch.float16)
        self.device = self.model.device

        self.stop_tokens = ["\n", "\n\n",".", ","]

    def query(self, msg):
        #input_ids = self.tokenizer([msg], return_tensors="pt").input_ids.to(self.device)
        input_ids = self.tokenizer([msg], return_tensors="pt").to(self.device)

        stopping_criteria = stop_sequences_criteria(
                self.tokenizer,
                self.stop_tokens,
                input_ids["input_ids"].shape[1],
                input_ids["input_ids"].shape[0],
            )
        with torch.no_grad():
            outputs = self.model.generate(input_ids.input_ids,
                temperature=self.temperature,
                max_new_tokens=self.max_output_tokens,
                stopping_criteria=stopping_criteria,
                eos_token_id=self.tokenizer.eos_token_id
                )
            input_length = input_ids['input_ids'].shape[1]
            generated_tokens = outputs[:, input_length:]
            generated_text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        # out = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # result = out[len(msg):]
        # return result
        return generated_text