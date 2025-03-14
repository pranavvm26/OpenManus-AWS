import json
from typing import Dict, List, Optional, Union

from openai import (
    APIError,
    AsyncAzureOpenAI,
    AsyncOpenAI,
    AuthenticationError,
    OpenAIError,
    RateLimitError,
)
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall  # Moved to top
from tenacity import retry, stop_after_attempt, wait_random_exponential
from litellm import acompletion  # Use async completion for litellm

from app.config import LLMSettings, config
from app.logger import logger
from app.schema import (
    ROLE_VALUES,
    TOOL_CHOICE_TYPE,
    TOOL_CHOICE_VALUES,
    Message,
    ToolChoice,
)

REASONING_MODELS = ["o1", "o3-mini"]

class LLM:
    _instances: Dict[str, "LLM"] = {}

    def __new__(
        cls, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ):
        if config_name not in cls._instances:
            instance = super().__new__(cls)
            instance.__init__(config_name, llm_config)
            cls._instances[config_name] = instance
        return cls._instances[config_name]

    def __init__(
        self, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ):
        if not hasattr(self, "client"):  # Only initialize if not already initialized
            llm_config = llm_config or config.llm
            llm_config = llm_config.get(config_name, llm_config["default"])
            self.model = llm_config.model
            self.max_tokens = llm_config.max_tokens
            self.temperature = llm_config.temperature
            self.api_type = llm_config.api_type
            self.api_key = llm_config.api_key
            self.api_version = llm_config.api_version
            self.base_url = llm_config.base_url

            # Initialize client based on api_type
            if self.api_type == "azure":
                self.client = AsyncAzureOpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key,
                    api_version=self.api_version,
                )
                self.use_litellm = False
            elif self.api_type == "openai":
                self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
                self.use_litellm = False
            elif self.api_type == "litellm":
                self.use_litellm = True
                # litellm handles Bedrock authentication via AWS credentials (e.g., boto3)
                # No explicit client initialization needed here
            else:
                raise ValueError(f"Unsupported api_type: {self.api_type}")

    @staticmethod
    def format_messages(messages: List[Union[dict, Message]]) -> List[dict]:
        """
        Format messages for LLM by converting them to OpenAI message format.
        (Unchanged from original)
        """
        formatted_messages = []

        for message in messages:
            if isinstance(message, dict):
                if "role" not in message:
                    raise ValueError("Message dict must contain 'role' field")
                formatted_messages.append(message)
            elif isinstance(message, Message):
                formatted_messages.append(message.to_dict())
            else:
                raise TypeError(f"Unsupported message type: {type(message)}")

        for msg in formatted_messages:
            if msg["role"] not in ROLE_VALUES:
                raise ValueError(f"Invalid role: {msg['role']}")
            if "content" not in msg and "tool_calls" not in msg:
                raise ValueError(
                    "Message must contain either 'content' or 'tool_calls'"
                )

        return formatted_messages

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
    )
    async def ask(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = True,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Send a prompt to the LLM and get the response.
        """
        try:
            # Format system and user messages
            if system_msgs:
                system_msgs = self.format_messages(system_msgs)
                messages = system_msgs + self.format_messages(messages)
            else:
                messages = self.format_messages(messages)

            params = {
                "model": self.model,
                "messages": messages,
            }

            if self.model in REASONING_MODELS:
                params["max_completion_tokens"] = self.max_tokens
            else:
                params["max_tokens"] = self.max_tokens
                params["temperature"] = temperature or self.temperature

            if self.use_litellm:
                # Handle Bedrock via litellm
                params["stream"] = stream
                if not stream:
                    response = await acompletion(**params)
                    if not response.choices or not response.choices[0].message.content:
                        raise ValueError("Empty or invalid response from LLM")
                    return response.choices[0].message.content

                # Streaming with litellm
                response = await acompletion(**params)
                collected_messages = []
                async for chunk in response:
                    chunk_message = chunk.choices[0].delta.content or ""
                    collected_messages.append(chunk_message)
                    print(chunk_message, end="", flush=True)

                print()  # Newline after streaming
                full_response = "".join(collected_messages).strip()
                if not full_response:
                    raise ValueError("Empty response from streaming LLM")
                return full_response

            else:
                # Handle OpenAI/Azure
                if not stream:
                    params["stream"] = False
                    response = await self.client.chat.completions.create(**params)
                    if not response.choices or not response.choices[0].message.content:
                        raise ValueError("Empty or invalid response from LLM")
                    return response.choices[0].message.content

                params["stream"] = True
                response = await self.client.chat.completions.create(**params)
                collected_messages = []
                async for chunk in response:
                    chunk_message = chunk.choices[0].delta.content or ""
                    collected_messages.append(chunk_message)
                    print(chunk_message, end="", flush=True)

                print()  # Newline after streaming
                full_response = "".join(collected_messages).strip()
                if not full_response:
                    raise ValueError("Empty response from streaming LLM")
                return full_response

        except ValueError as ve:
            logger.error(f"Validation error: {ve}")
            raise
        except OpenAIError as oe:
            logger.error(f"OpenAI API error: {oe}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ask: {e}")
            raise

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
    )
    async def ask_tool(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        timeout: int = 300,
        tools: Optional[List[dict]] = None,
        tool_choice: TOOL_CHOICE_TYPE = ToolChoice.AUTO,  # type: ignore
        temperature: Optional[float] = None,
        **kwargs,
    ):
        """
        Ask LLM using functions/tools and return the response.
        """
        try:
            # Validate tool_choice (only relevant for OpenAI)
            if tool_choice not in TOOL_CHOICE_VALUES:
                raise ValueError(f"Invalid tool_choice: {tool_choice}")

            # Format messages
            if system_msgs:
                system_msgs = self.format_messages(system_msgs)
                messages = system_msgs + self.format_messages(messages)
            else:
                messages = self.format_messages(messages)

            # Validate tools if provided
            if tools:
                for tool in tools:
                    if not isinstance(tool, dict) or "type" not in tool:
                        raise ValueError("Each tool must be a dict with 'type' field")

            # Set up the completion request
            params = {
                "model": self.model,
                "messages": messages,
                "timeout": timeout,
                **kwargs,
            }

            if self.model in REASONING_MODELS:
                params["max_completion_tokens"] = self.max_tokens
            else:
                params["max_tokens"] = self.max_tokens
                params["temperature"] = temperature or self.temperature

            # Add tools to params for OpenAI, but rely on system prompt for Bedrock
            if not self.use_litellm:
                params["tools"] = tools
                params["tool_choice"] = tool_choice
            else:
                params["drop_params"] = True  # Drop unsupported params for Bedrock

            # Call the LLM
            if self.use_litellm:
                response = await acompletion(**params)
                content = response.choices[0].message.content
                logger.debug(f"Bedrock response content: {content}")

                # Parse the response for tool invocation
                try:
                    # Attempt to parse as JSON for tool invocation
                    tool_data = json.loads(content)
                    if isinstance(tool_data, dict) and "tool" in tool_data:
                        logger.info(f"Tool invocation detected: {tool_data}")
                        return ChatCompletionMessage(
                            role="assistant",
                            content=None,
                            tool_calls=[
                                ChatCompletionMessageToolCall(
                                    id="call_1",  # Dummy ID
                                    type="function",
                                    function={
                                        "name": tool_data["tool"],
                                        "arguments": json.dumps({k: v for k, v in tool_data.items() if k != "tool"})
                                    }
                                )
                            ]
                        )
                except json.JSONDecodeError:
                    # If not JSON, treat as plain text response
                    pass

                # Return plain text response if no tool invocation
                return ChatCompletionMessage(role="assistant", content=content, tool_calls=None)

            else:
                # OpenAI/Azure path
                response = await self.client.chat.completions.create(**params)
                if not response.choices or not response.choices[0].message:
                    logger.error(f"Invalid response from LLM: {response}")
                    raise ValueError("Invalid or empty response from LLM")
                return response.choices[0].message

        except ValueError as ve:
            logger.error(f"Validation error in ask_tool: {ve}")
            raise
        except OpenAIError as oe:
            if isinstance(oe, AuthenticationError):
                logger.error("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.error("Rate limit exceeded. Consider increasing retry attempts.")
            elif isinstance(oe, APIError):
                logger.error(f"API error: {oe}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ask_tool: {e}")
            raise