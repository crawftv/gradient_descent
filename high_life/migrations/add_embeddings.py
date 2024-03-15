from high_life.settings import embed, chroma_collection

if __name__ == "__main__":
    nodes = chroma_collection.get()
    breakpoint()
    embeddings = embed(nodes["documents"])
    chroma_collection.update(ids=nodes["ids"], embeddings=embeddings)
