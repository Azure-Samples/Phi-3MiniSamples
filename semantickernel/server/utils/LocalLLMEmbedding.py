import os
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv


class LocalLLMEmbedding:
    def __init__(self, model_name):

        load_dotenv()

        self.model_name = model_name.lower()
        self.EMBEDDING_URL = os.environ.get("EMBEDDING_URL")
        self.tokenizer = AutoTokenizer.from_pretrained(self.EMBEDDING_URL)

        if self.model_name.find("jina")>-1:
            self.model = AutoModel.from_pretrained(self.EMBEDDING_URL, trust_remote_code=True) 
        
        if self.model_name.find("text2veccn")>-1:
            self.model = SentenceTransformer(self.EMBEDDING_URL)

        if self.model_name.find("bge")>-1:
            self.model = SentenceTransformer(self.EMBEDDING_URL)


    def call_local_llm_embeddings(self, sentences):

        encodings = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
        embeddings = self.model.encode(sentences)

        return embeddings, encodings.input_ids.numel()
