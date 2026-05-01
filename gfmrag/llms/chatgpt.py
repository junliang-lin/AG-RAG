import logging
import os
import time

import dotenv
import tiktoken
from openai import OpenAI

from .base_language_model import BaseLanguageModel

logger = logging.getLogger(__name__)
# Disable OpenAI and httpx logging
# Configure logging level for specific loggers by name
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

dotenv.load_dotenv()

os.environ["TIKTOKEN_CACHE_DIR"] = "./tmp"

# OPENAI_MODEL = ["gpt-4.1-mini", "gpt-4o-mini", "gpt-5-mini", "gpt-5-nano"]


def get_token_limit(model: str = "gpt-4") -> int:
    """Returns the token limitation of provided model"""
    if model in ["gpt-4.1-mini", "gpt-4o-mini", "gpt-5-mini", "gpt-5-nano"]:
        num_tokens_limit = 128000
    else:
        num_tokens_limit = 128000
    # else:
    #     raise NotImplementedError(
    #         f"""get_token_limit() is not implemented for model {model}."""
    #     )
    return num_tokens_limit


class ChatGPT(BaseLanguageModel):
    """A class that interacts with OpenAI's ChatGPT models through their API.

    This class provides functionality to generate text using ChatGPT models while handling
    token limits, retries, and various input formats.

    Args:
        model_name_or_path (str): The name or path of the ChatGPT model to use
        retry (int, optional): Number of retries for failed API calls. Defaults to 5

    Attributes:
        retry (int): Maximum number of retry attempts for failed API calls
        model_name (str): Name of the ChatGPT model being used
        maximun_token (int): Maximum token limit for the specified model
        client (OpenAI): OpenAI client instance for API interactions

    Methods:
        token_len(text): Calculate the number of tokens in a given text
        generate_sentence(llm_input, system_input): Generate response using the ChatGPT model

    Raises:
        KeyError: If the specified model is not found when calculating tokens
        Exception: If generation fails after maximum retries
    """

    def __init__(self, model_name_or_path: str, retry: int = 5):
        self.retry = retry
        self.model_name = model_name_or_path
        self.maximun_token = get_token_limit(self.model_name)

        client = OpenAI(
            api_key=os.environ[
                "OPENAI_API_KEY"
            ],  # this is also the default, it can be omitted
        )
        # client = OpenAI(
        #     base_url="https://openrouter.ai/api/v1",
        #     api_key="",
        # )
        self.client = client

    def token_len(self, text: str) -> int:
        """Returns the number of tokens used by a list of messages."""
        try:
            encoding = tiktoken.encoding_for_model('gpt-4o-mini')
            num_tokens = len(encoding.encode(text))
        except KeyError as e:
            raise KeyError(f"Warning: model {self.model_name} not found.") from e
        return num_tokens

    def generate_sentence(
        self, llm_input: str | list, system_input: str = ""
    ) -> str | Exception:
        """Generate a response using the ChatGPT API.

        This method sends a request to the ChatGPT API and returns the generated response.
        It handles both single string inputs and message lists, with retry logic for failed attempts.

        Args:
            llm_input (Union[str, list]): Either a string containing the user's input or a list of message dictionaries
                in the format [{"role": "role_type", "content": "message_content"}, ...]
            system_input (str, optional): System message to be prepended to the conversation. Defaults to "".

        Returns:
            Union[str, Exception]: The generated response text if successful, or the Exception if all retries fail.
                The response is stripped of leading/trailing whitespace.

        Raises:
            Exception: If all retry attempts fail, returns the last encountered exception.

        Notes:
            - Automatically truncates inputs that exceed the maximum token limit
            - Uses exponential backoff with 30 second delays between retries
            - Sets temperature to 0.0 for deterministic outputs
            - Timeout is set to 60 seconds per API call
        """

        # If the input is a list, it is assumed that the input is a list of messages
        if isinstance(llm_input, list):
            message = llm_input
        else:
            message = []
            if system_input:
                message.append({"role": "system", "content": system_input})
            message.append({"role": "user", "content": llm_input})
        cur_retry = 0
        num_retry = self.retry
        # Check if the input is too long
        message_string = "\n".join([m["content"] for m in message])
        input_length = self.token_len(message_string)
        if input_length > self.maximun_token:
            print(
                f"Input lengt {input_length} is too long. The maximum token is {self.maximun_token}.\n Right tuncate the input to {self.maximun_token} tokens."
            )
            llm_input = llm_input[: self.maximun_token]
        error = Exception("Failed to generate sentence")
        while cur_retry <= num_retry:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name, 
                    messages=message, 
                    timeout=60, 
                    temperature=1.0,
                    # top_p=0.95,
                    # presence_penalty=1.5,
                    # extra_body={
                    #     "top_k": 20,
                    # }, 
                    # temperature = 1.0,
                    reasoning_effort="low",
                )
                result = response.choices[0].message.content.strip()  # type: ignore
                if hasattr(response.usage, "completion_tokens_details") and response.usage.completion_tokens_details:
                    details = response.usage.completion_tokens_details
                    if hasattr(details, "reasoning_tokens") and details.reasoning_tokens:
                        thinking_tokens = details.reasoning_tokens
                    else:
                        thinking_tokens = 0
                # logger.info(f"input tokens: {response.usage.prompt_tokens}, output tokens: {response.usage.completion_tokens}, thinking tokens: {thinking_tokens}, total tokens: {response.usage.total_tokens}")
                token_statistics = {
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                    "thinking_tokens": thinking_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
                return result, token_statistics
            except Exception as e:
                logger.error("Message: ", llm_input)
                logger.error("Number of token: ", self.token_len(message_string))
                logger.error(e)
                time.sleep(30)
                cur_retry += 1
                error = e
                continue
        return error
