# 📘 GSM8K Evaluation: Upgraded Strategy & Usage Guide

This document explains the transition from a traditional **Few-Shot Prompting** approach to a more robust **Zero-Shot Chain-of-Thought (CoT)** evaluation for the GSM8K task. It also provides clear instructions on how to reproduce results using both **single-GPU** and **multi-GPU** environments.

> ⚠️ **Purpose**: To match previous dheyo's benchmarking scripts. [Original README.md](https://github.com/varunikanaini/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/README.md)

---

## 🔁 Overview of Changes

| **Aspect**                 | **Original (Few-Shot Prompting)**                        | **Updated (Zero-Shot CoT Prompting)**                                                            |
| -------------------------- | -------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| **Prompt (`doc_to_text`)** | `Question: {{question}}\nAnswer:`                        | Instructional persona prompt with reasoning steps and final answer format (`#### [Your Answer]`) |
| **Few-shot Examples**      | `num_fewshot: 5`                                         | `num_fewshot: 0`                                                                                 |
| **Decoding**               | Greedy decoding (`do_sample: false`, `temperature: 0.0`) | Sampling enabled (`do_sample: true`, `temperature: 0.6`, `top_p: 0.95`, `top_k: 40`)             |
| **Max Token Limit**        | Implicit (default)                                       | `max_new_tokens: 32678` — prevents truncation of step-by-step reasoning                          |
| **Instruction Clarity**    | Minimal (relies on examples)                             | Explicit instructions encourage consistent reasoning and output format                           |
| **Regex Filters**          | Extract numeric answers via `####`                       | Same regex filters maintained for backward compatibility                                         |

---

## ⚙️ Installation Prerequisite (For AMD GPUS)

> Before evaluation, install the nightly ROCm-compatible PyTorch stack and upgrade ggufs:

```bash
pip install --pre --upgrade torch torchvision torchao --index-url https://download.pytorch.org/whl/nightly/rocm6.4

pip install --upgrade gguf
```

---

## 🧪 Evaluation Commands

> Run the bash script

```bash
chmod +x run_eval.sh
./run_eval.sh
```

### ▶️ **Single-GPU Evaluation**

```bash
lm_eval --model hf \
  --model_args pretrained=${MODEL_DIR},gguf_file=${GGUF_FILE},tokenizer=${TOKENIZER_DIR} \
  --tasks gsm8k \
  --device cuda:0 \
  --batch_size 1 \
  --output_path ${OUTPUT_DIR} \
  --log_samples
```

### ⚡ **Multi-GPU Evaluation**

```bash
accelerate launch --multi_gpu --num_processes=${NUM_GPUS} -m lm_eval \
  --model hf \
  --model_args pretrained=${MODEL_DIR},gguf_file=${GGUF_FILE},tokenizer=${TOKENIZER_DIR} \
  --tasks gsm8k \
  --device cuda:0 \
  --batch_size 4 \
  --output_path ${OUTPUT_DIR} \
  --log_samples
```

📝 **Logs and sample generations** are saved under `result_logs/`

---

## 🧾 YAML Configuration Diff (Before vs After)

Only the **relevant parts of the YAML file** have been modified. See below:

```diff
- doc_to_text: "Question: {{question}}\nAnswer:"
+ doc_to_text: |
+   You are a helpful math assistant. Solve the given math problem step by step...
+   Always end with: #### [Your Answer]
+   Solve: {{question}}

- num_fewshot: 5
+ num_fewshot: 0

- do_sample: false
+ do_sample: true

- temperature: 0.0
+ temperature: 0.6

+ top_p: 0.95
+ top_k: 40

+ max_new_tokens: 32678
```

Everything else (e.g., `doc_to_target`, `regexes_to_ignore`, filters) remains the same to ensure compatibility with previous evaluation logic.

---

## 📌 Final Notes

* This configuration more closely reflects **Dheyo's Benchmarking Scripts**.
* The changes ware being tested with the model: `DeepSeek-R1-Distill-Qwen-1.5B-gsm8k_v8_qat.gguf`.

---

