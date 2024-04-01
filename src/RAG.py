from src.knowledge_base import KnowledgeBase
from src.language_model import LanguageModel


class RAG:

    def __init__(self):
        self.llm = LanguageModel()

        self.knowledge_base = KnowledgeBase('data/example.pdf')
        self.knowledge_base.build_dataset()
        self.knowledge_base.encode()

    def forward(self, prompt):
        knowlege_base_entry = self.knowledge_base.search(prompt)
        context = [x['text'] for x in knowlege_base_entry]
        response = self.llm.generate(context, prompt)
        print(f"Context: {knowlege_base_entry}")
        print(f"Response: {response}")

