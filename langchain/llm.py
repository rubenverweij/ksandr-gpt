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
        response = llm.invoke(prompt, stream=False)
        logging.info(f"The text is: {chunk}")
        logging.info(f"The summary is: {response}")
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

    def first_n_words(self, n=4000):
        matches = list(re.finditer(r"\S+", self.text))
        if len(matches) <= n:
            return self.text
        return self.text[: matches[n - 1].end()]

    def summarize(self, len_chunk_sum: int = 400, len_final_sum: int = 200) -> str:
        chunks = self.chunk_text(self.text)
        chunk_lengths = self.calculate_chunk_summary_length(chunks[:2], len_chunk_sum)
        summaries = [
            self.summarize_chunk(chunk, length)
            for chunk, length in zip(chunks[:1], chunk_lengths)
        ]
        # Finalize the summary
        # llm = self.llm_manager.get_llm()
        final_summary = ". ".join(summaries)
        # response = llm.invoke(
        #     self.template.format(words=len_final_sum, tekst=final_summary)
        # )
        # logging.info(f"The complete summarised text is: {final_summary}")
        # logging.info(f"The summary is: {response}")
        # if isinstance(response, dict):
        #     if "choices" in response:
        #         return response["choices"][0].get("text", "").strip()
        #     elif "content" in response:
        #         return response["content"].strip()
        return final_summary

    def summarize_simple(self, len_chunk_sum: int = 500) -> str:
        if len(self.text) < 1000:
            return "De text is te kort om te kunnen samenvatten."
        if isinstance(self.text, str):
            summary = self.summarize_chunk(
                chunk=self.trim_context_to_fit(), summary_length=len_chunk_sum
            )
            return summary
        return "Kan geen samenvatting maken van bestand."

    def count_tokens(self, text: str) -> int:
        return len(self.llm_manager.get_llm().tokenize(text.encode("utf-8")))

    def trim_context_to_fit(self) -> str:
        n_ctx = 4096
        max_tokens = 500
        # Build a prompt with an empty context just to measure overhead
        llm = self.llm_manager.get_llm()
        dummy_prompt = self.template.format(words=max_tokens, tekst="")
        prompt_overhead_tokens = self.count_tokens(dummy_prompt)
        available_tokens_for_context = n_ctx - max_tokens - prompt_overhead_tokens
        context_tokens = llm.tokenize(self.text.encode("utf-8"))
        if len(context_tokens) > available_tokens_for_context:
            trimmed_tokens = context_tokens[
                -available_tokens_for_context:
            ]  # Keep latest context
            input_text_summary = llm.detokenize(trimmed_tokens).decode(
                "utf-8", errors="ignore"
            )
        return input_text_summary
