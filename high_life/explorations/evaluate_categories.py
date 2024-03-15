import pandas

from high_life.settings import chroma_collection

x = {"ids": [], "metadatas": [], "documents": [], "embeddings": []}
for i in range(8):
    for key, value in chroma_collection.get(limit=1000, offset=i * 1000,
                                            include=["metadatas", "documents", "embeddings"]).items():
        if key in x and value is not None:
            values = x.get(key)
            values.extend(value)
            x[key] = values

df = pandas.DataFrame(x)
