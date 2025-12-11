import gc
import threading
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import LlamaCpp
from typing import List
import re
import logging

logging.basicConfig(level=logging.INFO)


class LLMManager:
    def __init__(self, model_path, max_tokens, temperature, top_p, n_gpu_layers):
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.n_gpu_layers = n_gpu_layers

        self.lock = threading.Lock()
        self.llm = None
        self.current_ctx = None

    def load_llm(self, n_ctx: int):
        """Load a llama.cpp model with the given n_ctx."""
        with self.lock:
            # Unload previous model
            if self.llm is not None:
                del self.llm
                self.llm = None
                gc.collect()

            # Load new model
            self.llm = LlamaCpp(
                model_path=self.model_path,
                max_tokens=self.max_tokens,
                n_ctx=n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                temperature=self.temperature,
                top_p=self.top_p,
                streaming=False,
                verbose=False,
            )

            self.current_ctx = n_ctx

    def get_llm(self):
        if self.llm is None:
            raise RuntimeError("LLM not loaded yet.")
        return self.llm


class RecursiveSummarizer:
    def __init__(self, llm_manager, text, template):
        self.llm_manager = llm_manager
        self.text = text
        self.template = template

    def count_words(self, text: str) -> int:
        return len(re.findall(r"\b[\w’'-]+\b", text, flags=re.UNICODE))

    def chunk_text(self, text: str) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max(self.llm_manager.current_ctx // 2, 5000),
            chunk_overlap=50,
        )
        return splitter.split_text(text)

    def summarize_chunk(self, chunk: str, summary_length: int) -> str:
        llm = self.llm_manager.get_llm()
        prompt = self.template.format(words=summary_length, tekst=chunk)
        logging.info(
            f"LLM loaded and prompt formatted for chunk {len(chunk)} with summary len {summary_length}"
        )
        response = llm.invoke(prompt)
        if isinstance(response, dict):
            if "choices" in response:
                return response["choices"][0].get("text", "").strip()
            elif "content" in response:
                return response["content"].strip()
        return str(response).strip()

    def calculate_chunk_summary_length(
        self, chunks: List[str], final_words: int
    ) -> List[int]:
        total_words = sum(self.count_words(c) for c in chunks)
        return [
            max(1, int(self.count_words(c) / total_words * final_words)) for c in chunks
        ]

    def summarize(self, final_words: int = 800) -> str:
        chunks = self.chunk_text(self.text)
        chunk_lengths = self.calculate_chunk_summary_length(chunks, final_words)
        summaries = [
            self.summarize_chunk(chunk, length)
            for chunk, length in zip(chunks, chunk_lengths)
        ]
        return " ".join(summaries)
