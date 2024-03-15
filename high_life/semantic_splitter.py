if __name__ == "__main__":
    # embed_model = HuggingFaceEmbedding(
    #     model_name="BAAI/bge-small-en-v1.5"
    # )
    from bs4 import BeautifulSoup

    with open('data/download.html', 'r') as f:
        # Read the contents of the file into a Beautiful Soup document
        doc = BeautifulSoup(f, 'html')
        breakpoint()
        for p in doc.find_all('p'):
            print(f"P:{p.text}")


class Chunk:
    def __init__(self, chunk, llm):
        self.chunk = chunk
        self.llm = llm

    @property
    def summary(self):
        return self.llm.complete(f"Summarize the text below:\n{self.chunk.text}").text
