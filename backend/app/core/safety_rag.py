import os
import pathway as pw
from typing import Any, cast
from pathway.xpacks.llm.llms import LiteLLMChat
from pathway.xpacks.llm.embedders import LiteLLMEmbedder
from pathway.xpacks.llm.splitters import TokenCountSplitter
from pathway.xpacks.llm.document_store import DocumentStore
from pathway.stdlib.indexing.nearest_neighbors import BruteForceKnnFactory  # FIXED: was pathway.xpacks.llm.indexing.nearest_neighbors
from pathway.engine import BruteForceKnnMetricKind
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL    = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct:free")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
SAFETY_DOCS_DIR     = os.getenv("SAFETY_DOCS_DIR", "./safety_docs")


def build_rag_pipeline():
    """
    Builds the Pathway RAG pipeline with correct API.

    Architecture:
    ┌────────────────────────────────────────────────────────┐
    │              PATHWAY LLM xPack RAG                     │
    │                                                        │
    │  ./safety_docs/*.txt                                   │
    │         │                                              │
    │         ▼                                              │
    │   pw.io.fs.read()       ← Live streaming source        │
    │   mode=streaming — auto-detects file add/change/delete │
    │         │                                              │
    │         ▼                                              │
    │   TokenCountSplitter    ← Chunks docs to 300 tokens    │
    │         │                                              │
    │         ▼                                              │
    │   LiteLLMEmbedder       ← HuggingFace all-MiniLM       │
    │   (free, local, 384-dim vectors)                       │
    │         │                                              │
    │         ▼                                              │
    │   BruteForceKnnFactory  ← Retriever factory            │
    │   (correct Pathway API — NOT embedder= on DocStore)    │
    │         │                                              │
    │         ▼                                              │
    │   DocumentStore(docs, retriever_factory, splitter)     │
    │   ← Live BM25 + KNN index, always up-to-date          │
    │         │                                              │
    │         ▼                                              │
    │   doc_store.retrieve(query, k=3)                       │
    │   ← Returns top-3 relevant safety rule chunks          │
    │         │                                              │
    │         ▼                                              │
    │   LiteLLMChat via OpenRouter                           │
    │   ← Generates report grounded in retrieved rules       │
    └────────────────────────────────────────────────────────┘

    KEY API NOTES (common mistakes to avoid):
      ✅ DocumentStore(docs, retriever_factory=..., splitter=...)
      ❌ DocumentStore(docs, embedder=...)   # WRONG — no embedder param

      ✅ BruteForceKnnFactory(reserved_space, embedder, metric, dimensions)
      ❌ passing embedder directly to DocumentStore

      ✅ LiteLLMChat(model="openai/mistral-...", api_base=OPENROUTER_BASE_URL)
      ❌ LiteLLMChat.achat()   # no achat method — use litellm.acompletion() directly

      ✅ doc_store.retrieve(query=..., k=3)
      ❌ rag.answer(question)  # BaseRAGQuestionAnswerer is a server class,
                               # not a simple callable
    """

   
    documents = pw.io.fs.read(
        SAFETY_DOCS_DIR,
        format="binary",
        mode="streaming",
        with_metadata=True,
    )

   
    embedder = LiteLLMEmbedder(
        capacity=5,
        retry_strategy=None,
        cache_strategy=None,
        model="huggingface/sentence-transformers/all-MiniLM-L6-v2",
    )

   
    retriever_factory = BruteForceKnnFactory(
        reserved_space=1000,
        embedder=embedder,
        metric=BruteForceKnnMetricKind.COSINE,  # Use enum for metric  # type: ignore[attr-defined]
        dimensions=384,
    )


    doc_store: Any = DocumentStore(
        docs=documents,
        retriever_factory=retriever_factory,   # ← correct param name
        splitter=TokenCountSplitter(max_tokens=300),
    )

    llm = LiteLLMChat(
        model=f"openai/{OPENROUTER_MODEL}",
        api_key=OPENROUTER_API_KEY,
        api_base=OPENROUTER_BASE_URL,
        temperature=0.3,
        max_tokens=400,
    )

    return doc_store, llm



if __name__ == "__main__":
    import asyncio
    import litellm

    os.makedirs(SAFETY_DOCS_DIR, exist_ok=True)
    test_doc = os.path.join(SAFETY_DOCS_DIR, "osha_builtin_rules.txt")
    if not os.path.exists(test_doc):
        with open(test_doc, "w") as f:
            f.write("OSHA Rule: All workers must wear hard hats at all times.\n"
                    "OSHA Rule: Fire detection requires immediate evacuation.\n"
                    "OSHA Rule: Risk score above 8 requires halting operations.\n")

    print("[TEST] Building RAG pipeline...")
    doc_store, llm = build_rag_pipeline()

    async def test():
       
        results = doc_store.retrieve(
            query="worker without helmet construction site violation",
            k=3,
        )
    
        chunks = [r["text"] if isinstance(r, dict) and "text" in r else "" for r in results]
        context = "\n".join(chunks)
        print(f"[TEST] Retrieved context:\n{context}\n")

        response = await litellm.acompletion(  
            model=f"openai/{OPENROUTER_MODEL}",
            api_key=OPENROUTER_API_KEY,
            api_base=OPENROUTER_BASE_URL,
            messages=[{
                "role": "user",
                "content": f"Safety rules: {context}\n\nQuestion: 3 workers without helmets in Zone C. What should supervisor do?"
            }],
            max_tokens=200,
        )
        resp_dict = cast(Any, response)
        print(f"[TEST] Answer:\n{resp_dict['choices'][0]['message']['content']}")  

    asyncio.run(test())