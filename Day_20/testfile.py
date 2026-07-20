# test_10_conversations.py
# ============================================
# STEP 6: 10 multi-turn conversations test karna
# Har conversation live API (/api/rag/chat) ko call karta hai.
# Result ek JSON file mein save hota hai - documentation likhne ke liye.
# ============================================

import requests
import json
import time

API_URL = "http://localhost:8001/api/rag/chat"
CONVERSATIONS_FILE = "testfile.json"


def load_conversations(path: str) -> list:
    """conversations.json file se test conversations load karta hai -
    isay yahan hardcode nahi kiya, taake conversations ko edit karne ke
    liye Python code na chhedna pade, sirf JSON file update karni ho."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_conversation(conv):
    session_id = f"test_conv_{conv['id']}"
    print(f"\n{'='*70}")
    print(f"Conversation {conv['id']}: {conv['label']}")
    print('='*70)

    turn_results = []
    for turn_num, message in enumerate(conv["turns"], 1):
        print(f"\n  Turn {turn_num} - User: {message}")
        try:
            response = requests.post(
                API_URL,
                json={"session_id": session_id, "message": message},
                timeout=60,
            )
            data = response.json()
            print(f"  Assistant: {data['answer'][:200]}...")
            turn_results.append({
                "turn": turn_num,
                "question": message,
                "answer": data.get("answer", ""),
                "sources": data.get("sources", []),
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            turn_results.append({
                "turn": turn_num,
                "question": message,
                "answer": f"ERROR: {e}",
                "sources": [],
            })

        time.sleep(1)  # thora rukna, taake API pe load na ho

    return {
        "conversation_id": conv["id"],
        "label": conv["label"],
        "session_id": session_id,
        "turns": turn_results,
    }


if __name__ == "__main__":
    print("Starting 10-conversation test suite...")
    print("Make sure main.py server is running on http://localhost:8001\n")

    conversations = load_conversations(CONVERSATIONS_FILE)

    all_results = []
    for conv in conversations:
        result = run_conversation(conv)
        all_results.append(result)

    # Results ko file mein save karo - documentation likhne ke liye
    output_path = "test_results_10_conversations.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n\n{'='*70}")
    print(f"DONE. Poore results save ho gaye: {output_path}")
    print("Ab is file ko dekh kar documentation likhein - kahan memory ne")
    print("help ki, kahan noise ban gaya.")
    print('='*70)