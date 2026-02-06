import gc
import threading
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import LlamaCpp
from typing import List
import logging
from helpers import detect_concluding_chunk, verwijder_onafgeronde_zinnen

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
        summary_len = 300
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
        return verwijder_onafgeronde_zinnen(response)

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
