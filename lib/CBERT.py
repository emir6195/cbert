
from ragatouille import RAGPretrainedModel


class CBERT:
    def __init__(self) -> None:
        self.RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

    def train(self, docs: list[str], index_name: str, meta_datas : list[dict], overwrite_index=True) -> bool:
        try : 
            self.RAG.index(
                index_name=index_name,
                collection=docs,
                document_metadatas=meta_datas
            )
            return True
        except Exception as e :
            print(e)
            return False
    
    def invoke(self, index_name: str, query: str) : 
        try: 
            return self.RAG.search(
                query=query,
                index_name=index_name,
                k=4
            )
        except Exception as e :
            print(e)
            return False
        

cbert = CBERT()