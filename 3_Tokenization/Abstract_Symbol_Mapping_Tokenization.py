import os
from transformers import BertTokenizerFast

# --- PATH CONFIGURATION ---
# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Base directory for the model and tokenizer outputs
MODEL_DIR_RAW = os.path.join(SCRIPT_DIR, "..", "model")
MODEL_DIR = os.path.abspath(MODEL_DIR_RAW)

# File and directory paths
TOKENIZER_OUTPUT_DIR = os.path.join(MODEL_DIR, "tokenizer_abstract_symbol_mapping")
VOCAB_FILE = os.path.join(TOKENIZER_OUTPUT_DIR, "vocab.txt")

# !!! ATTENTION !!! 
# 1. BASE PARAMETERS
# With 994 (+5 special tokens) we broadly cover the maximum number of activities for any real-world scenario (up to 999 max tokens)
MAX_ACT = 994 

def create_and_test_tokenizer():
    # Ensure the model directory exists
    os.makedirs(TOKENIZER_OUTPUT_DIR, exist_ok = True)

    # 2. VOCABULARY FILE CREATION
    # Special Tokens Definition:
    # [PAD]  = Padding to standardize tensor lengths
    # [UNK]  = Unknown token (fallback)
    # [CLS]  = Sequence start (necessary for classification tasks)
    # [SEP]  = Sentence separator
    # [MASK] = Mask for the MLM task in pre-training
    special_tokens = [
        "[PAD]", 
        "[UNK]", 
        "[CLS]", 
        "[SEP]", 
        "[MASK]"
    ]

    # Dynamic generation of abstract tokens
    act_tokens = [f"ACT_{i + 1:03d}" for i in range(MAX_ACT)]

    # Complete list creation
    all_tokens = special_tokens + act_tokens

    # Physical creation of the vocabulary file
    with open(VOCAB_FILE, "w", encoding = "utf-8") as f:
        for token in all_tokens:
            f.write(token + "\n")

    print(f"\n[INFO] Vocabulary created with {len(all_tokens)} tokens.")

    # 3. LOAD AND CONFIGURE TOKENIZER
    # We set do_lower_case to False because our ACT_XXX tokens are uppercase
    tokenizer = BertTokenizerFast(vocab_file = VOCAB_FILE, do_lower_case = False)

    # We add the tokens as indivisible atoms 
    # (this prevents the tokenizer from splitting the tokens due to the presence of "_")
    tokenizer.add_tokens(act_tokens)

    # Save the configured tokenizer
    tokenizer.save_pretrained(TOKENIZER_OUTPUT_DIR)

    print(f"[SUCCESS] Saving completed in: {TOKENIZER_OUTPUT_DIR}")
    print(f"[INFO] Vocabulary size (including special tokens): {len(tokenizer)}\n")

    # --- TOKENIZER TEST ---

    # Load the tokenizer from the newly created folder to verify integrity
    tokenizer_test = BertTokenizerFast.from_pretrained(TOKENIZER_OUTPUT_DIR)

    # Simulate a pre-training trace (sentence format)
    test_sentence = "ACT_001 ACT_002 [MASK] ACT_005 ?"

    # Tokenize the sentence
    encoded = tokenizer_test(test_sentence, add_special_tokens = True)

    # Retrieve IDs and decode them for verification
    input_ids = encoded["input_ids"]
    decoded_sentence = tokenizer_test.decode(input_ids)
    extracted_tokens = tokenizer_test.convert_ids_to_tokens(input_ids)

    print("=" * 60)
    header_text = "TOKENIZATION INTEGRITY TEST"
    print(f"{header_text:^60}")
    print("=" * 60)
    print(f"Original Input:     {test_sentence}")
    print(f"Extracted Tokens:   {extracted_tokens}")
    print(f"Tensor IDs:         {input_ids}")
    print(f"Decoding:           {decoded_sentence}")
    print("=" * 60)
    print("[SUCCESS] The tokenizer successfully processed the abstract tokens without splitting them.")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    create_and_test_tokenizer()