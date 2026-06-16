import json
import os

def fix_jsonl(filepath: str) -> None:
    """
    Reads the jsonl file, converts system+user+assistant messages into user+assistant,
    and overwrites the file.
    
    Args:
        filepath (str): Path to the jsonl file to modify.
    """
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
        
    fixed_items = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            messages = item.get("messages", [])
            
            # Reconstruct messages to alternate user/assistant
            new_messages = []
            system_content = ""
            
            for msg in messages:
                role = msg.get("role")
                content = msg.get("content", "")
                
                if role == "system":
                    system_content = content
                elif role == "user":
                    if system_content:
                        # Prepend system prompt to the user message
                        new_content = f"{system_content}\n\n{content}"
                        new_messages.append({"role": "user", "content": new_content})
                        system_content = "" # reset
                    else:
                        new_messages.append({"role": "user", "content": content})
                elif role == "assistant":
                    new_messages.append({"role": "assistant", "content": content})
                    
            item["messages"] = new_messages
            fixed_items.append(item)
            
    with open(filepath, "w", encoding="utf-8") as f:
        for item in fixed_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"Successfully fixed {len(fixed_items)} items in {filepath}!")

if __name__ == "__main__":
    fix_jsonl("data/raft_training_pairs.jsonl")
