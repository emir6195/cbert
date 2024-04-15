
from ragatouille import RAGPretrainedModel


class CBERT:
    def __init__(self) -> None:
        self.RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
        self.cached_indexes = {}

    def train(self, docs: list[str], index_name: str, meta_datas: list[dict], overwrite_index=True) -> bool:
        try:
            print("index_name =>", index_name)
            print("docs =>", len(docs))
            print("overwrite_index =>", overwrite_index)
            if overwrite_index:
                self.RAG.index(
                    index_name=index_name,
                    collection=docs,
                    document_metadatas=meta_datas,
                )
                return True
            else:
                self.RAG.add_to_index(
                    index_name=index_name,
                    new_collection=docs,
                    new_document_metadatas=meta_datas,
                )
        except Exception as e:
            print(e)
            return False

    def invoke(self, index_name: str, query: str):
        try:
            # if (index_name not in self.cached_indexes):
            model = self.RAG.from_index(
                ".ragatouille/colbert/indexes/"+index_name)
            self.cached_indexes[index_name] = model
            return model.search(
                query=query,
                k=4
            )
            # else:
            #     self.cached_indexes[index_name].search(
            #         query=query,
            #         k=4
            #     )
        except Exception as e:
            print(e)
            return False


cbert = CBERT()
