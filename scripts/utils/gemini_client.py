import asyncio
import time
import random
from typing import Any, Dict, List, Optional
import structlog
import httpx

logger = structlog.get_logger()

class GeminiClient:
    """
    Production-grade rate-limited Gemini API client supporting multi-key rotation,
    exponential backoff, shared connection pooling, and robust error recovery.
    """
    def __init__(self, api_keys: List[str], rpm_limit: int = 15, max_retries: int = 5):
        self.api_keys = api_keys
        self.rpm_limit = rpm_limit
        self.max_retries = max_retries
        self.current_key_idx = 0
        self.spacing = 60.0 / rpm_limit
        self.last_request_times = [0.0] * len(api_keys)
        self.lock = asyncio.Lock()
        self.client = httpx.AsyncClient(timeout=60.0)
        
    async def get_next_key_and_wait(self) -> str:
        async with self.lock:
            idx = self.current_key_idx
            self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
            
            now = time.monotonic()
            elapsed = now - self.last_request_times[idx]
            wait_time = max(0.0, self.spacing - elapsed)
            # Reserve the slot optimistically inside the lock
            self.last_request_times[idx] = now + wait_time
            
        # Sleep outside the lock so other coroutines can acquire the lock
        if wait_time > 0:
            await asyncio.sleep(wait_time)
        return self.api_keys[idx]

    async def generate_content(
        self,
        model: str,
        system_instruction: str,
        user_content: str,
        temperature: float = 0.4,
        json_mode: bool = False
    ) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": user_content}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": temperature
            }
        }
        
        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [
                    {"text": system_instruction}
                ]
            }
            
        if json_mode:
            payload["generationConfig"]["responseMimeType"] = "application/json"
            
        headers = {
            "Content-Type": "application/json"
        }
        
        for attempt in range(self.max_retries):
            api_key = await self.get_next_key_and_wait()
            params = {"key": api_key}
            
            try:
                response = await self.client.post(url, json=payload, params=params, headers=headers)
                
                if response.status_code == 200:
                    res_json = response.json()
                    candidates = res_json.get("candidates", [])
                    if not candidates:
                        raise ValueError(f"Gemini API returned no candidates: {res_json}")
                    
                    part = candidates[0].get("content", {}).get("parts", [])[0]
                    text = part.get("text", "")
                    return text
                    
                elif response.status_code == 429:
                    logger.warning(
                        "gemini_api_rate_limited",
                        attempt=attempt+1,
                        status_code=response.status_code,
                        body=response.text[:200]
                    )
                else:
                    logger.warning(
                        "gemini_api_error_status",
                        attempt=attempt+1,
                        status_code=response.status_code,
                        body=response.text[:200]
                    )
            except Exception as e:
                logger.warning("gemini_api_connection_error", attempt=attempt+1, error=str(e))
                
            # Exponential backoff with random jitter
            backoff = (2 ** attempt) + random.uniform(0.1, 1.0)
            logger.info("gemini_api_backing_off", seconds=backoff)
            await asyncio.sleep(backoff)
            
        raise RuntimeError(f"Failed to get response from Gemini API after {self.max_retries} attempts.")

    async def close(self):
        await self.client.aclose()
