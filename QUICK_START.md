"""
QUICK START GUIDE: DBHS Fine-Tuning Pipeline v2

This guide walks you through the complete workflow.
"""

# ============================================================
# STEP 0: INSTALL DEPENDENCIES
# ============================================================

"""
If using the .venv virtualenv:

    cd f:\Codes\DBHS Intern
    .\.venv\Scripts\python.exe -m pip install --upgrade pip
    .\.venv\Scripts\python.exe -m pip install \
        torch \
        datasets \
        transformers \
        peft \
        trl \
        numpy

Alternatively, run in one line:

    .\.venv\Scripts\python.exe -m pip install torch datasets transformers peft trl numpy
"""

# ============================================================
# STEP 1: ANALYZE DATA QUALITY (5 minutes)
# ============================================================

"""
COMMAND:
    .\.venv\Scripts\python.exe analyze_data_quality.py

WHAT IT DOES:
- Shows dataset statistics (size, Q&A length distribution)
- Finds duplicate questions
- Identifies repeated phrases
- Flags overrepresented content (e.g., "Michael Krieger")
- Shows quality issues and recommendations

WHAT TO LOOK FOR:
- If duplicate_pairs > 100, dataset has significant redundancy
- If long_answers > 50, consider truncating
- If "Michael" appears >5%, content is heavily skewed

EXAMPLE OUTPUT:
    [✓] Loaded 2752 examples
    Total examples: 2752
    Exact duplicate questions: 42
    Duplicate Q&A pairs: 42
    'Michael': 234 examples (8.5%)
    ✗ 'Michael' appears >5% of examples; content is heavily skewed
"""

# ============================================================
# STEP 2: TRAIN NEW MODEL (30-45 minutes on GPU)
# ============================================================

"""
COMMAND:
    .\.venv\Scripts\python.exe train_v2.py

WHAT IT DOES:
1. Loads train.jsonl
2. Deduplicates dataset
3. Formats using Qwen2.5-Instruct chat template
4. Applies LoRA adapter
5. Trains for 3 epochs with validation
6. Saves adapter to dbhs_lora_v2/

EXPECTED OUTPUT:
    [*] Loading tokenizer...
    [*] Chat template available: True
    [*] Loading model...
    [*] LoRA applied. Trainable params: 8,388,608
    [*] Loading dataset...
    [*] Loaded 2752 examples
    [*] After deduplication: 2752 → 2710 examples (removed 42 duplicates)
    [*] Formatting dataset...
    [*] Example formatted training text:
    ============================================================
    <|im_start|>user
    What is IB?
    <|im_end|>
    <|im_start|>assistant
    The International Baccalaureate (IB)...
    <|im_end|>
    ============================================================
    [*] Starting training...
    Training: 100%|████████████| 2015/2015 [XX:XX<00:00, 0.75it/s]
    [✓] Training complete. Adapter saved to dbhs_lora_v2/

WHAT TO MONITOR:
- Training loss should decrease each epoch
- Eval loss should decrease (not increase, which = overfitting)
- No CUDA errors or out-of-memory

TROUBLESHOOTING:
- If CUDA out of memory: Reduce per_device_train_batch_size in train_v2.py
- If training very slow: Check GPU utilization (nvidia-smi)
- If loss doesn't decrease: Check learning_rate in SFTConfig

OUTPUTS:
- dbhs_lora_v2/adapter_model.safetensors (the fine-tuned weights)
- dbhs_lora_v2/adapter_config.json (LoRA configuration)
- dbhs_lora_v2/tokenizer.json (tokenizer from Qwen2.5)
"""

# ============================================================
# STEP 3: TEST INFERENCE (Interactive Mode)
# ============================================================

"""
COMMAND:
    .\.venv\Scripts\python.exe main_v2.py

WHAT IT DOES:
- Loads Qwen2.5-0.5B-Instruct + LoRA adapter
- Starts interactive chat loop
- Type questions, model responds

INTERACTION:
    [*] Loading tokenizer from Qwen/Qwen2.5-0.5B-Instruct...
    [*] Loading base model Qwen/Qwen2.5-0.5B-Instruct...
    [*] Loading LoRA adapter from dbhs_lora_v2...
    
    ============================================================
    DBHS Information Assistant (Type 'quit' to exit)
    ============================================================
    
    You: What is IB?
    
    [Generating response...]
    
    Assistant: The International Baccalaureate (IB) is a rigorous...
    
    You: Tell me about bell schedules
    
    [Generating response...]
    
    Assistant: Our school has several bell schedules...
    
    You: quit
    Goodbye!

TESTING CHECKLIST:
✓ Response is relevant to the question
✓ No repeating text ("Michael Krieger Michael Krieger...")
✓ Uses proper sentence structure
✓ Answers actually from training data (not hallucinated)
✓ Format doesn't include "### Human:" or other artifacts
✓ Inference time is reasonable (~2-5 sec per query)

WHAT TO TRY:
1. "What is IB?"
2. "Tell me about Brahma Tech Academy"
3. "What are the bell schedules?"
4. "What clubs are available?"
5. "How do I get a work permit?"

COMPARE TO OLD MODEL:
- Old: Often ignored question, output unrelated text
- New: Should directly address the question
"""

# ============================================================
# STEP 4: COMMAND-LINE INFERENCE
# ============================================================

"""
Instead of interactive mode, you can query directly:

COMMAND:
    .\.venv\Scripts\python.exe main_v2.py "What is the Wellness Center?"

OUTPUTS:
    The Wellness Center is located in Room 254 and was opened in January 2016...

USE CASES:
- Integrate into web app
- Batch processing multiple questions
- API server response
"""

# ============================================================
# STEP 5: OPTIONAL - ITERATE & IMPROVE
# ============================================================

"""
If quality is not satisfactory:

OPTION A: Try larger model
- Edit train_v2.py: MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
- Rerun training (slower, but better quality)

OPTION B: Improve data quality
- Run analyze_data_quality.py
- Manually review top repeated phrases / "Michael Krieger" examples
- Remove low-quality examples from train.jsonl
- Retrain

OPTION C: Adjust generation parameters
- Edit main_v2.py
- Modify temperature, top_p, repetition_penalty
- Retest inference

OPTION D: Longer training
- Edit train_v2.py: num_train_epochs = 5 (or more)
- Allow more iterations over dataset

OPTION E: Different LoRA settings
- Edit lora_config in train_v2.py
- Increase r=32 (more LoRA capacity)
- Adjust lora_alpha
"""

# ============================================================
# TROUBLESHOOTING
# ============================================================

"""
PROBLEM: CUDA errors during training
SOLUTION:
- Reduce per_device_train_batch_size to 2 or 1
- Reduce MAX_SEQ_LENGTH to 1024

PROBLEM: Inference very slow (>10 sec per query)
SOLUTION:
- Check GPU utilization: nvidia-smi
- Try smaller model: Qwen/Qwen2.5-0.5B (already using this)
- Use CPU if inference batch size is small

PROBLEM: Model outputs complete gibberish
SOLUTION:
- Check that dbhs_lora_v2/ exists
- Check that training completed successfully
- Try a known good query: "What is IB?"
- Compare vs. old model output

PROBLEM: Model repeats same phrase multiple times
SOLUTION:
- Increase repetition_penalty (currently 1.2, try 1.5 or 2.0)
- Decrease temperature (currently 0.5, try 0.3)

PROBLEM: Model doesn't know about something in training data
SOLUTION:
- Run analyze_data_quality.py to check if it's in dataset
- Model may need more training (increase num_train_epochs)
- If rare question, may not have learned it (expected)
"""

# ============================================================
# FILES CREATED
# ============================================================

"""
train_v2.py
  - Corrected training pipeline
  - Uses Qwen2.5-0.5B-Instruct
  - Deduplication + proper chat template
  - Run: python train_v2.py

main_v2.py
  - Corrected inference pipeline
  - Fixed generation settings
  - Proper format consistency
  - Run: python main_v2.py

analyze_data_quality.py
  - Dataset analysis tool
  - Shows duplicates, repeated phrases, etc.
  - Run: python analyze_data_quality.py

REFACTOR_EXPLANATION.md
  - Detailed documentation
  - Explains every change and why
  - Reference for understanding the improvements

This file (QUICK_START.md)
  - Quick reference guide
  - Step-by-step instructions
"""

# ============================================================
# EXPECTED IMPROVEMENTS
# ============================================================

"""
BEFORE (using distilgpt2 + old pipeline):
- Input: "Hello"
- Output: "Dr. Michael B. Krieger, Ph.D...." (unrelated)
- Problem: Format mismatch + model too small + training errors

AFTER (using Qwen2.5 + corrected pipeline):
- Input: "Hello"
- Output: "Hello! I'm the DBHS Information Assistant. Ask me about..."
- Improvement: Proper format + larger model + clean training

BEFORE (asking about bell schedules):
- Output: Ignored question, output corporate boilerplate

AFTER:
- Output: "Our regular bell schedule is... [accurate times]"
- Improvement: Model understands question context

BEFORE (repeated queries):
- Same prompt → different outputs each time (instability)
- Problem: Overtraining on duplicates

AFTER:
- Same prompt → consistent output (reproducible)
- Improvement: Deduplication + larger model capacity
"""
