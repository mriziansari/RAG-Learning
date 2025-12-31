import numpy as np
import uuid
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions

from create_documents import DocumentCreator
from hierarchical_chuking import HierarchicalChunker


class EmbeddingManager:
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
      self.model_name = model_name
      self.model = None
      self._load_model()
       

    def _load_model(self):
      """Loading the SentenceTransformer model"""  
      try:
        print(f"Loading model {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        print(f"Model {self.model_name} loaded successfully. Embedding Dimentions: {self.model.get_sentence_embedding_dimension()}")
      except Exception as e:
        raise Exception(f"Failed to load model {self.model_name}: {e}")
      
       
    def generate_embedding(self, texts: list[str]) -> list[np.ndarray]:
      try:
        if not self.model:
          raise Exception("Model not loaded. Please load the model first.") 
        print(f"Generating embedding for {len(texts)} texts")
        print("-" * 100)
        genrated_embeddings = self.model.encode(texts,
         show_progress_bar=True,
         convert_to_numpy=True,
         normalize_embeddings=True
         )
        print(f"Generated embedding for {len(texts)} texts with shape {genrated_embeddings.shape}")
        print("-" * 100)
        return genrated_embeddings

      except Exception as e:
        raise Exception(f"Failed to generate embedding: {e}")

class VectorStoreManager:
  def __init__(self, persist_path: str):
    self.client = chromadb.PersistentClient(path=persist_path)

  def reset_vector_collection(self, name: str):
    try:
      self.client.delete_collection(name)
    except Exception:
      pass
    
  def add_chunks_to_collection(
    self, 
    collection_name: str, 
    chunks,
    embeddings,
    id_prefix: str
    ):
    try:
      collection = self.client.get_or_create_collection(collection_name)
      collection.add(
        documents=[chunk.page_content for chunk in chunks],
        embeddings=embeddings.tolist(),
        metadatas=[chunk.metadata for chunk in chunks],
        ids=[f"{id_prefix}_{i}" for i in range(len(chunks))] 
      )

      print(f"Successfully added {len(chunks)} chunks to the collection {collection_name}")

    except Exception as e:
      raise Exception(f"Failed to add chunks to the collection {name}: {e}")

def main():
  doc_path = "urban_infrastrucher.pdf"
  doc_id = "urban_infrastrucher_v1"
  model_name = "all-mpnet-base-v2"
  VECTOR_PATH = "data/urban_infrastructure"

# Loading the document
  document_creator = DocumentCreator(doc_path)
  documents = document_creator.read_pdf()
# Chunks
  hierarchical_chunker = HierarchicalChunker()
  router_chunks, answer_chunks = hierarchical_chunker.create_chunks(documents,doc_id)
# Embeddings
  embedding_manager = EmbeddingManager(model_name)
  router_embeddings = embedding_manager.generate_embedding([chunk.page_content for chunk in router_chunks])
  answer_embeddings = embedding_manager.generate_embedding([chunk.page_content for chunk in answer_chunks])

# Vector Store
  vector_store_manager = VectorStoreManager(VECTOR_PATH)
  
  vector_store_manager.reset_vector_collection("router_collection")
  vector_store_manager.reset_vector_collection("answer_collection")

  vector_store_manager.add_chunks_to_collection(
    "router_collection", 
    router_chunks, 
    router_embeddings, 
    "router")
  vector_store_manager.add_chunks_to_collection(
    "answer_collection", 
    answer_chunks, 
    answer_embeddings, 
    "answer")

if __name__ == "__main__":
  main()

  
  
  
  