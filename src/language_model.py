import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel


class LanguageModel:

    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.eval()

    def generate(self, context, query):
        prompt = f"Context: {context}. \n Question: {query} \n Answer: ".replace('., ', '. ').replace('\'', '').replace(']', '').replace('[', '')

        x = self.tokenizer.encode(prompt, return_tensors='pt')
        mask = torch.ones(x.shape, dtype=torch.long)

        output = self.model.generate(
            input_ids=x,
            attention_mask=mask,
            pad_token_id=self.tokenizer.eos_token_id,
            max_length=264,
            temperature=1.0,
            num_return_sequences=1,
        )

        return self.tokenizer.decode(output[0], skip_special_tokens=False)[len(prompt):].split('Question')[0]
