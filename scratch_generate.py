import asyncio
import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from backend.database.connection import init_db_pool, get_db_connection, close_db_pool
from ollama import AsyncClient
from backend.config import OLLAMA_HOST

async def prototype_generation():
    print("Initializing DB pool...")
    await init_db_pool()
    
    ollama_client = AsyncClient(host=OLLAMA_HOST)
    
    async with get_db_connection() as conn:
        # Get one target node
        row = await conn.fetchrow(
            """
            SELECT node_id, doc_id, title, text_content, breadcrumb
            FROM ast_nodes
            WHERE text_content IS NOT NULL 
              AND needs_repair = FALSE 
              AND length(text_content) > 150
            LIMIT 1 OFFSET 10
            """
        )
        if not row:
            print("No suitable target node found.")
            return
            
        target = dict(row)
        print("Target node selected:")
        print(f"ID: {target['node_id']}")
        
    system_prompt = (
        "You are a dataset generator. You return a JSON object with exactly two keys: 'question' and 'answer'.\n"
        "The value of 'question' is a string.\n"
        "The value of 'answer' MUST be a single string containing the exact markdown formatted text with headers, table, and source quotes. Do NOT make the value of 'answer' a JSON object or dictionary.\n"
        "Example output format:\n"
        "{\n"
        "  \"question\": \"What are the capital requirements for an IBU?\",\n"
        "  \"answer\": \"**Short Answer:** The minimum capital is USD 20 million.\\n\\n| Rule | Detail | Source |\\n|---|---|---|\\n| Capital Requirement | Minimum USD 20 million | Section 3(a) |\\n\\n**Plain Language:** An IBU must have at least USD 20 million in capital to operate.\\n\\n**Source Quote:** \\\"The minimum capital requirement for an IFSC Banking Unit shall be USD 20 million.\\\"\"\n"
        "}"
    )

    user_prompt = (
        "Based on the following target clause, generate one compliance question and its answer string.\n\n"
        "TARGET CLAUSE:\n"
        f"Breadcrumb: {target['breadcrumb']}\n"
        f"Title: {target['title']}\n"
        f"Content: {target['text_content']}\n\n"
        "Ensure the generated answer uses ONLY facts from the target clause.\n"
        "Return a JSON object with keys 'question' (string) and 'answer' (string)."
    )
    
    print("\nCalling Ollama (llama3.2:3b)...")
    try:
        response = await ollama_client.chat(
            model="llama3.2:3b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            format="json",
            options={"temperature": 0.3}
        )
        content_str = response.message.content
        print("\nResponse from Ollama:")
        print(content_str)
        
        parsed = json.loads(content_str)
        print("\nParsed successfully!")
        print(f"Generated Question: {parsed.get('question')}")
        print(f"Generated Answer Type: {type(parsed.get('answer'))}")
        print(f"Generated Answer:\n{parsed.get('answer')}")
        
    except Exception as e:
        print("Failed to generate or parse:", e)
        
    await close_db_pool()

if __name__ == "__main__":
    asyncio.run(prototype_generation())
