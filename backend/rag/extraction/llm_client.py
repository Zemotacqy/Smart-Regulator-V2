import json
from typing import Type, List, Dict, Any
from pydantic import BaseModel, ValidationError
from ollama import AsyncClient
import structlog
from backend.config import OLLAMA_HOST, EMBEDDING_MODEL

logger = structlog.get_logger()

# Global AsyncClient instance
_client = AsyncClient(host=OLLAMA_HOST)

async def generate_embedding(text: str) -> List[float]:
    """
    Generates a dense vector embedding using nomic-embed-text.
    Ollama context length parameter (num_ctx) is explicitly set to 8192.
    """
    try:
        response = await _client.embeddings(
            model=EMBEDDING_MODEL,
            prompt=text,
            options={"num_ctx": 8192}
        )
        return response.get("embedding", [])
    except Exception as e:
        logger.error("embedding_generation_failed", model=EMBEDDING_MODEL, error=str(e))
        raise

class LLMClientError(Exception):
    """Base exception for LLM client operations."""
    pass

class LLMValidationError(LLMClientError):
    """Raised when the LLM output fails validation after self-healing."""
    pass

async def call_llm_with_validation(
    model: str,
    messages: List[Dict[str, str]],
    response_schema: Type[BaseModel],
    temperature: float = 0.0,
    keep_alive: int = 10,
    num_ctx: int = None
) -> BaseModel:
    """
    Calls Ollama with JSON formatting and validates the response against a Pydantic schema.
    If validation fails, performs 1 self-heal attempt by providing the error back to the model.
    
    Args:
        model: The name of the model to call.
        messages: Conversation history.
        response_schema: The Pydantic model class to validate the JSON against.
        temperature: Sampling temperature (default 0.0 for deterministic outputs).
        keep_alive: Time in seconds to keep the model loaded in memory after the request.
        
    Returns:
        An instance of response_schema containing the validated data.
        
    Raises:
        LLMValidationError: If the response cannot be parsed or validated after self-healing.
        LLMClientError: For general Ollama API issues.
    """
    options = {"temperature": temperature}
    if num_ctx is not None:
        options["num_ctx"] = num_ctx
    
    # First attempt
    try:
        logger.debug("llm_call_attempt_1", model=model, messages_count=len(messages))
        response = await _client.chat(
            model=model,
            messages=messages,
            format="json",
            options=options,
            keep_alive=keep_alive
        )
        content = response.message.content
        logger.debug("llm_response_1", content=content)
        
        # Parse and validate
        return response_schema.model_validate_json(content)
        
    except (ValidationError, json.JSONDecodeError, Exception) as e:
        logger.warning(
            "llm_validation_failed_attempt_1",
            model=model,
            error=str(e),
            raw_content=locals().get("content", None)
        )
        
        # Prepare self-heal message
        err_msg = str(e)
        self_heal_messages = list(messages)
        
        # If we got a response content, append it to show the model what it wrote
        if "content" in locals() and content:
            self_heal_messages.append({"role": "assistant", "content": content})
        
        self_heal_messages.append({
            "role": "user",
            "content": (
                f"YOUR PREVIOUS RESPONSE FAILED VALIDATION WITH ERROR:\n{err_msg}\n\n"
                f"Please correct the JSON output. Do NOT write any conversational prefix or suffix. "
                f"Respond with a valid JSON object matching the schema."
            )
        })
        
        # Second attempt (self-heal)
        try:
            logger.info("llm_call_self_heal_attempt", model=model, messages_count=len(self_heal_messages))
            response = await _client.chat(
                model=model,
                messages=self_heal_messages,
                format="json",
                options=options,
                keep_alive=keep_alive
            )
            heal_content = response.message.content
            logger.debug("llm_response_self_heal", content=heal_content)
            
            return response_schema.model_validate_json(heal_content)
            
        except Exception as heal_err:
            logger.error(
                "llm_self_heal_failed",
                model=model,
                error=str(heal_err),
                raw_content=locals().get("heal_content", None)
            )
            raise LLMValidationError(
                f"Failed to validate LLM output after 1 self-heal attempt. Original error: {err_msg}. "
                f"Heal error: {str(heal_err)}"
            ) from heal_err
