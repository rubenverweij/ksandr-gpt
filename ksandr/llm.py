"""
This module defines the LLMManager class for managing the lifecycle, loading,
and configuration of Large Language Models (LLMs) via llama.cpp.

It provides thread-safe routines for model loading, context management, and
LLM access, enabling integration with the Ksandr API for Dutch infrastructure
question answering, summarization, and prompt-based workflows.

Core functionality includes:
- Loading and unloading llama.cpp models with dynamic context sizes (n_ctx)
- Parameterized instantiation of models (temperature, top_p, etc)
- Locking for concurrent safe access
- Utilities for returning configured LLM objects

Dependencies: langchain_community.llms (LlamaCpp), helpers (text post-processing functions), threading, gc
"""

import gc
import threading
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import LlamaCpp
from typing import List
import logging
from helpers import detect_concluding_chunk, remove_unfinished_sentences

logging.basicConfig(level=logging.INFO)


class LLMManager:
    """
    LLMManager is responsible for managing the lifecycle and configuration of Large Language Models (LLMs)
    instantiated via llama.cpp within the Ksandr project.

    Key features:
    - Thread-safe loading and reloading of models, supporting dynamic changes in context window (`n_ctx`)
    - Safe access to the active LLM via internal locking mechanism
    - Parameterization for temperature, top_p, GPU usage, and inference settings
    - Unloading of previous models with explicit garbage collection to conserve memory

    Usage:
      - Initialize with model parameters (path, max_tokens, temperature, etc.)
      - Call `load_llm(n_ctx)` to (re)load the model with a specified context size
      - Use `get_llm()` to retrieve the current loaded model for inference tasks

    Exceptions:
      - Raises RuntimeError if inference is attempted before a model is loaded

    This class enables resource-efficient management of LLMs in production APIs requiring low latency,
    high concurrency, and rapid context swapping.
    """

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
                top_k=40,
                repetition_penalty=1.2,
                # stop=["\n\n", "###"],
            )

            self.current_ctx = n_ctx

    def get_llm(self):
        if self.llm is None:
            raise RuntimeError("LLM not loaded yet.")
        return self.llm


class RecursiveSummarizer:
    """
    Class for recursively summarizing long documents by chunking text and aggregating partial summaries.

    This class provides utilities to:
        - Split long texts into manageable chunks using a recursive text splitter.
        - Generate concise summaries of each chunk using a provided LLMManager and prompt templates.
        - Aggregate and refine these partial summaries into a final, summarized output.

    Args:
        llm_manager: The LLMManager instance used to invoke the LLM.
        text (str): The text to be summarized.
        template_initial (str): Prompt template for the initial summary chunk(s).
        template_partial (str): Prompt template for iterative (partial) summaries.
        template_conclude (str): Prompt template for concluding the summary.
        template_correction (str): Prompt template for correcting summaries.
        template_full (str): Prompt template for full summary generation.
        template_single (str): Prompt template for summarizing a single chunk.

    Methods:
        chunk_text(text): Splits the input text into chunks suitable for the LLM context window.
        summarize_chunk(text, template): Returns a summary of the input text chunk using the LLM and a specified template.
        summarize(): Produces a complete summary of the input text, suitable for use in downstream applications or as a user-facing result.
    """

    def __init__(
        self,
        llm_manager,
        text,
        template_initial,
        template_partial,
        template_conclude,
        template_correction,
        template_full,
        template_single,
    ):
        self.llm_manager = llm_manager
        self.text = text
        self.template_initial = template_initial
        self.template_partial = template_partial
        self.template_conclude = template_conclude
        self.template_correction = template_correction
        self.template_full = template_full
        self.template_single = template_single

    def chunk_text(self, text: str) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max(self.llm_manager.current_ctx // 2, 9_000),
            chunk_overlap=50,
        )
        return splitter.split_text(text)

    def summarize_chunk(self, text: str, template: str) -> str:
        llm = self.llm_manager.get_llm()
        summary_len = 400
        prompt = template.format(text=text, words=summary_len)
        response = llm.invoke(prompt, stream=False)
        logging.info(f"The summary is: {response}")
        if isinstance(response, dict):
            if "choices" in response:
                return response["choices"][0].get("text", "").strip()
            elif "content" in response:
                return response["content"].strip()
        return str(response).strip()

    def summarize(self) -> str:
        """Make a summary of the document.

        Args:
            len_chunk_sum (int, optional): _description_. Defaults to 400.
            len_final_sum (int, optional): _description_. Defaults to 200.

        Returns:
            str: _description_
        """

        chunks = self.chunk_text(self.text)
        logging.info(f"Starting summarizing {len(chunks)}")

        # If there are multiple chunks
        summaries = []
        if len(chunks) > 1:
            summaries.append(
                self.summarize_chunk(chunks[0], template=self.template_initial)
            )

            # Check for summary chunk
            chunks_with_conclusions = detect_concluding_chunk(chunks)
            if len(chunks_with_conclusions) == 0:
                chunks = list(dict.fromkeys(chunks[:1] + chunks[-1:]))
                summaries.append(
                    self.summarize_chunk(chunks[1], template=self.template_partial)
                )
            else:
                logging.info("Found chunk with conclusions")
                summaries.append(
                    self.summarize_chunk(
                        chunks_with_conclusions[0], template=self.template_conclude
                    )
                )

            # Start creating final summary:
            final_text = ". ".join(summaries)
            response = self.summarize_chunk(final_text, template=self.template_full)

        else:
            # In case there is only one chunk
            response = self.summarize_chunk(chunks[0], template=self.template_single)

        logging.info(f"The final summary is: {response}")

        response = self.summarize_chunk(response, template=self.template_correction)

        logging.info(f"The corrected summary is: {response}")
        return remove_unfinished_sentences(response)

    def summarize_simple(self) -> str:
        """
        Summarize the document using a simplified approach.

        Returns:
            str: The generated summary, or an error message if summarization is not possible.
        """
        if len(self.text) < 1000:
            return "De text is te kort om te kunnen samenvatten."
        if isinstance(self.text, str):
            summary = self.summarize_chunk(text=self.trim_context_to_fit())
            return summary
        return "Kan geen samenvatting maken van bestand."

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the given text using the language model's tokenizer.

        Args:
            text (str): The input text to be tokenized.

        Returns:
            int: The number of tokens in the input text.
        """
        return len(self.llm_manager.get_llm().client.tokenize(text.encode("utf-8")))

    def trim_context_to_fit(self) -> str:
        """
        Trims the input text to ensure it fits within the model's context window,
        accounting for the prompt template and a specified number of output tokens.

        Returns:
            str: The input text, trimmed if necessary to fit within the allowed token budget.
        """
        n_ctx = 4000
        max_tokens = 500
        # Build a prompt with an empty context just to measure overhead
        input_text_summary = self.text
        llm = self.llm_manager.get_llm().client
        dummy_prompt = self.template.format(words=max_tokens, tekst="")
        prompt_overhead_tokens = self.count_tokens(dummy_prompt)
        available_tokens_for_context = n_ctx - max_tokens - prompt_overhead_tokens
        context_tokens = llm.tokenize(self.text.encode("utf-8"))
        if len(context_tokens) > available_tokens_for_context:
            trimmed_tokens = context_tokens[
                :available_tokens_for_context
            ]  # Keep latest context
            input_text_summary = llm.detokenize(trimmed_tokens).decode(
                "utf-8", errors="ignore"
            )
        return input_text_summary
