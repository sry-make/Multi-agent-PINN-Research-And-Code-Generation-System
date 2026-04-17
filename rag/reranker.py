"""
BGE Reranker — 从 V1 迁移，升级至 GPU 推理

变更:
    - device 默认改为 "cuda"（RTX 4060 Ti）
    - 接口与 V1 完全兼容，直接替换
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder
from config import (
    RERANKER_MODEL_NAME,
    RERANKER_MODEL_SOURCE,
    RERANKER_DEVICE,
)


class BGEReranker:
    """BGE 交叉编码器重排器（V2: GPU 加速）"""

    _instance: BGEReranker | None = None  # 模型复用，避免重复加载

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def _load(self):
        if self._loaded:
            return
        print(f"🔄 加载 Reranker: {RERANKER_MODEL_NAME} (device={RERANKER_DEVICE})")
        self.model = CrossEncoder(RERANKER_MODEL_SOURCE, device=RERANKER_DEVICE)
        self._loaded = True
        print("✅ Reranker 加载完成")

    def rerank(
        self,
        query: str,
        documents: list[str],
        metadatas: list[dict],
        top_k: int = 5,
    ) -> tuple[list[str], list[dict]]:
        """对 (query, doc) 打分并返回精排后的 Top-K"""
        self._load()

        if not documents:
            return [], []

        pairs  = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(scores, documents, metadatas),
            key=lambda x: x[0],
            reverse=True,
        )[:top_k]

        return (
            [doc  for _, doc, _    in ranked],
            [meta for _, _,   meta in ranked],
        )
