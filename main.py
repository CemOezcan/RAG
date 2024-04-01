from src.RAG import RAG


def main():
    rag = RAG()
    while True:
        prompt = input('Ask a question!')
        if prompt == 'exit':
            exit()
        else:
            rag.forward(prompt)


if __name__ == '__main__':
    main()