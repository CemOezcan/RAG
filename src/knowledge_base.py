import faiss
from transformers import AutoTokenizer, AutoModel

import pdfplumber
import torch


class KnowledgeBase:

    def __init__(self, path):
        self.path = path
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model.eval()

        self.chunks = list()
        self.index = None

    def build_dataset(self):
        with pdfplumber.open(self.path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                line = 0
                for sentence in text.split('.\n'):
                    line += 1
                    line += sentence.count('\n')
                    if len(sentence) >= 45:
                        self.chunks.append({"text": sentence.replace('\n', ' '), "page": page_num, "line": line})

    @torch.no_grad()
    def forward(self, x):
        inputs = self.tokenizer(x, return_tensors="pt", padding='max_length', truncation=False, max_length=264)
        outputs = self.model(**inputs)

        return outputs.last_hidden_state[:, 0, :]

    def encode(self):
        embeddings = [self.forward(chunk['text']) for chunk in self.chunks]
        embeddings = torch.cat(embeddings, dim=0)

        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def search(self, prompt, k=1):
        query = self.forward(prompt)
        _, I = self.index.search(query, k)

        return [self.chunks[I[0][i]] for i in range(len(I[0]))]
