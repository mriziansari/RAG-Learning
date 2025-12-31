import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter


class HierarchicalChunker:
    def __init__(self):
        # Router chunks (parents)
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            separators=["\n## ", "\n### ", "\n\n", "\n"]
        )

        # Answer chunks (children)
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=60,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def _stable_section_id(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()[:10]

    def create_chunks(self, documents, doc_id: str):
        """
        Returns:
        - router_chunks (parents, level=1)
        - answer_chunks (children, level=2)
        """

        router_chunks = []
        answer_chunks = []

        parent_chunks = self.parent_splitter.split_documents(documents)

        for p_index, parent in enumerate(parent_chunks):
            section_id = self._stable_section_id(parent.page_content)

            # ---- Parent (Router Only) ----
            parent.metadata.update({
                "doc_id": doc_id,
                "level": 1,
                "section_id": section_id,
                "chunk_index": p_index,
                "role": "router"
            })

            router_chunks.append(parent)

            # ---- Children (Answer Only) ----
            child_chunks = self.child_splitter.split_documents([parent])

            for c_index, child in enumerate(child_chunks):
                child.metadata.update({
                    "doc_id": doc_id,
                    "level": 2,
                    "parent_section": section_id,
                    "chunk_index": c_index,
                    "role": "answer"
                })

                answer_chunks.append(child)

        return router_chunks, answer_chunks
