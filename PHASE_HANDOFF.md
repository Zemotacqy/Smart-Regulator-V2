# PHASE HANDOFF: PHASE 3 Fine-Tuning & Evaluation

This document outlines the final completion status of Phase 3 (Fine-Tuning & Evaluation) of the Smart Regulator RAG System, detailing components built, environment configurations, final evaluation results, resolved items, and deferred items.

---

## 1. PHASE 3 COMPLETION STATUS

- **Model Quantization & GGUF Conversion**: Updated `scripts/convert_to_gguf.sh` to resolve parameter binding issues (`--adapter-path`) and dynamically clones `llama.cpp` to resolve the `ModuleNotFoundError: No module named 'conversion'` error. Fused SaulLM adapters with the base model, converted to GGUF, and quantized to `Q4_K_M` format.
- **Ollama Deployment**: Successfully deployed and registered the fine-tuned model `ifsca-saullm-7b-ft:latest` in Ollama using `modelfiles/Modelfile.saullm`.
- **RAG Pipeline & LLM-as-a-Judge Evaluation**: Ran the full 91-question evaluation runner (`tests/run_eval_judge.py`) using `mistral-nemo:12b` as the judge model.
- **Compressor & Faithfulness Fixes**: Fixed the `ifsca-extractor-3b` timeouts by processing layout node compression sequentially (semaphore = 1) with an increased 25-second limit. Resolved alternative JSON schema generation issues (where `sentences` was returned instead of `relevant_sentences`) by adding a Pydantic `@model_validator` to `CompressorOutput` to map key variants properly. This successfully restored dropped contexts, improving faithfulness and citation accuracy.

---

## 2. FILES CREATED OR MODIFIED

- `/Users/manish/Downloads/repos/smart-regulator-v2/scripts/convert_to_gguf.sh` — Fixed parameters and added automated llama.cpp cloning to prevent module imports errors.
- `/Users/manish/Downloads/repos/smart-regulator-v2/AGENT_NOTES.md` — Updated with the final evaluation scores, resolved compressor timeouts details, and dense-retrieval limitations.
- `/Users/manish/Downloads/repos/smart-regulator-v2/PHASE_HANDOFF.md` — Documented Phase 3 completion status and handoff notes.

---

## 3. ENVIRONMENT STATE

- **PostgreSQL**: Local server active on `5432` with `smart_regulator_v2` database containing 1,752 AST nodes.
- **Ollama Service**: Local server active on `11434` with the fine-tuned model `ifsca-saullm-7b-ft:latest` registered and running.
- **Models Directory**:
  - LoRA adapters: `models/ifsca-saullm-7b-ft-adapters/`
  - Base model: `models/Saul-7B-Instruct-v1-4bit/`
  - Quantized model: `models/ifsca-saullm-7b-ft.Q4_K_M.gguf`

---

## 4. KNOWN ISSUES OR DEFERRED ITEMS

- **Low Dense Retrieval Recall (Recall@10 = 73.63%)**: The baseline combination of `nomic-embed-text:v1.5` dense vector search and PostgreSQL FTS BM25 does not reach the target Recall@10 of 92%. Legal cross-reference and glossary lookup queries require higher-precision dense/sparse matching. Recommend upgrading to `BAAI/bge-m3` or a legal-fine-tuned bi-encoder model.
- **Subsection Title Length Audit**: The title length checking in `auditor.py` is currently only active for `node_type == "SECTION"`. This check should be extended to cover `SUBSECTION` type nodes.
- **Breadcrumb Uniqueness**: Breadcrumbs for subclauses and clauses that lack explicit headings are not guaranteed to be unique within a section. Node UUIDs are currently used to guarantee citation mapping precision, but unique breadcrumbs remain a visual enhancement item.
