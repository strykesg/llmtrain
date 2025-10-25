#!/usr/bin/env python3
"""
Script to download and convert FinLang investopedia dataset
for market+news topic to JSONL format compatible with training.
"""

from datasets import load_dataset
import json
import random

def convert_to_chat_format(entry):
    """Convert a dataset entry to chat format."""
    # System message for financial/market analysis
    system_msg = "You are an elite Financial Analyst with decades of experience in financial markets. You provide comprehensive, deeply analytical responses that synthesize technical analysis, fundamental data, macroeconomic factors, and market sentiment."

    # User message is the question
    user_msg = entry['Question'].strip()

    # Assistant message is the answer
    assistant_msg = entry['Answer'].strip()

    # Format as chat completion
    chat_entry = {
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg}
        ]
    }

    return chat_entry

def main():
    print("Loading FinLang investopedia dataset...")

    # Load the dataset
    dataset = load_dataset('FinLang/investopedia-instruction-tuning-dataset', split='train')

    print(f"Full dataset size: {len(dataset)}")

    # Filter for market+news topic
    market_news_entries = [entry for entry in dataset if entry['Topic'] == 'market+news']

    print(f"Market+news entries found: {len(market_news_entries)}")

    # Optionally sample a subset (e.g., 10,000 entries) for manageable size
    # For now, let's take all of them but maybe sample if too large
    SAMPLE_SIZE = 50000  # Adjust as needed

    if len(market_news_entries) > SAMPLE_SIZE:
        print(f"Sampling {SAMPLE_SIZE} entries from {len(market_news_entries)} total...")
        market_news_entries = random.sample(market_news_entries, SAMPLE_SIZE)

    print(f"Converting {len(market_news_entries)} entries to chat format...")

    # Convert to chat format and save
    output_file = "/Users/bradleymutemi/Documents/llmtrain/secondary_data.jsonl"

    with open(output_file, 'w', encoding='utf-8') as f:
        for i, entry in enumerate(market_news_entries):
            try:
                chat_entry = convert_to_chat_format(entry)
                f.write(json.dumps(chat_entry, ensure_ascii=False) + '\n')

                if (i + 1) % 1000 == 0:
                    print(f"Processed {i + 1}/{len(market_news_entries)} entries...")

            except Exception as e:
                print(f"Error processing entry {i}: {e}")
                continue

    print(f"Conversion complete! Saved to: {output_file}")

    # Validate the output
    print("Validating output file...")
    valid_count = 0
    total_count = 0

    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            total_count += 1
            try:
                data = json.loads(line.strip())
                if 'messages' in data and len(data['messages']) == 3:
                    roles = [msg.get('role') for msg in data['messages']]
                    if roles == ['system', 'user', 'assistant']:
                        valid_count += 1
            except:
                continue

    print(f"Validation: {valid_count}/{total_count} entries are valid ({100*valid_count/total_count:.1f}%)")

    # Get file size
    import os
    file_size = os.path.getsize(output_file)
    print(f"File size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")

if __name__ == "__main__":
    main()
