# Copyright (c) Microsoft. All rights reserved.

import os
import torch

from transformers import AutoModelForCausalLM, AutoModel,AutoTokenizer
from transformers.generation.utils import GenerationConfig

# from . import InferenceGenerator

from dotenv import load_dotenv

# The model used to get the tokenizer can be a little arbitrary
# since the tokenizers are common within the same model type




# class CompletionGenerator(InferenceGenerator.InferenceGenerator):
class LocalLLMCompletion:
    def __init__(self,model_name):

        # super().__init__(model_name)

        load_dotenv()

        self.model_name = model_name.lower()

        self.CHAT_COMPLETION_URL = os.environ.get("CHAT_COMPLETION_URL")

        
        self.init_local_llm_model(model_name)


    


    def call_local_llm_chat(self, prompt, context, max_tokens,temperature):


        if self.model_name.find("chatglm3") > -1:
            history = []
            completion_text, _ = self.model.chat(self.tokenizer, prompt,  max_length=max_tokens,history=history,temperature=temperature)
        elif self.model_name.find("phi-2") > -1:
            inputs = self.tokenizer(prompt, return_tensors="pt",  return_attention_mask=False)
            outputs = self.model.generate(**inputs, max_length=max_tokens)
            completion_text = self.tokenizer.batch_decode(outputs)[0]
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        elif self.model_name.find("phi-3-mini") > -1:

            # torch.set_default_device("cpu")
            # inputs = self.tokenizer(prompt, return_tensors="pt",  return_attention_mask=False)
            messages = [
                {"role": "system", "content": "You are a AI assistant."},
                {"role": "user", "content": prompt},
            ]

            template = "{% for message in messages %}{{'<|' + message['role'] + '|>' + '\n' + message['content'] + '<|end|>\n' }}{% endfor %}"
            self.tokenizer.chat_template = template
            tokenized_chat = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
            outputs = self.model.generate(tokenized_chat, max_new_tokens=max_tokens, eos_token_id=32007)  # 32007 corresponds to <|end|>
            completion_text = self.tokenizer.batch_decode(outputs)[0]

            # outputs = self.model.generate(**inputs, max_length=max_tokens)
            # completion_text = self.tokenizer.batch_decode(outputs)[0]
        elif self.model_name.find("baichuan2") > -1:
            messages = []
            messages.append({"role": "user", "content": prompt})
            completion_text = self.model.chat(self.tokenizer, messages) 
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        else:
            completion_text = 'Sorry, unable to perform sentence completion with model'


        return completion_text
    
    def init_local_llm_model(self, model_name):

        if self.model_name.find("chatglm3") > -1:

            if torch.backends.mps.is_available():

                self.model = AutoModel.from_pretrained(self.CHAT_COMPLETION_URL, trust_remote_code=True).to("mps")

            elif torch.backends.cuda.is_available():

                self.model = AutoModel.from_pretrained(self.CHAT_COMPLETION_URL, trust_remote_code=True).half().cuda()

            else:
                self.model = AutoModel.from_pretrained(self.CHAT_COMPLETION_URL, trust_remote_code=True)

            
            self.tokenizer = AutoTokenizer.from_pretrained(self.CHAT_COMPLETION_URL, trust_remote_code=True)    

        if self.model_name.find("phi-2") > -1:
            
            self.model = AutoModelForCausalLM.from_pretrained(self.CHAT_COMPLETION_URL, trust_remote_code=True)


            self.tokenizer = AutoTokenizer.from_pretrained(self.CHAT_COMPLETION_URL, trust_remote_code=True)   

        if self.model_name.find("phi-3-mini") > -1:

            self.model = AutoModelForCausalLM.from_pretrained(self.CHAT_COMPLETION_URL, torch_dtype="auto", trust_remote_code=True)
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.CHAT_COMPLETION_URL, trust_remote_code=True)

        if self.model_name.find("baichuan2") > -1:
            self.model = AutoModelForCausalLM.from_pretrained(self.CHAT_COMPLETION_URL, torch_dtype=torch.float16,device_map="auto",trust_remote_code=True)


            self.tokenizer = AutoTokenizer.from_pretrained(self.CHAT_COMPLETION_URL,use_fast=False, trust_remote_code=True)
            # self.model = AutoModelForCausalLM.from_pretrained(self.CHAT_COMPLETION_URL, trust_remote_code=True).to("mps")
            self.model.generation_config = GenerationConfig.from_pretrained(self.CHAT_COMPLETION_URL)
