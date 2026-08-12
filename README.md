# Lightweight Retrieval-Augmented Generation (RAG) System
### Multimodal Document Intelligence with Open-Weight and Proprietary LLM Interchangeability

A self-contained, single-notebook implementation of a Retrieval-Augmented Generation pipeline that ingests unstructured and semi-structured enterprise documents (PDF, DOCX, XLSX), indexes them locally using open-weight embedding models, and answers natural-language questions — via text or speech — through a provider-agnostic language model layer supporting both a proprietary API (Google Gemini 2.5 Flash) and a fully open-weight alternative (Qwen2.5-7B-Instruct, via the Hugging Face Inference API).

The system is deployed as an interactive web application using [Gradio](https://www.gradio.app/), enabling rapid prototyping and stakeholder demonstration without frontend engineering overhead.

---

## 1. Overview

Large Language Models (LLMs) are trained on broad, internet-scale corpora and, as a result, lack grounded knowledge of an organization's proprietary or recent information. Retrieval-Augmented Generation (Lewis et al., 2020) addresses this limitation by retrieving relevant passages from a domain-specific knowledge base at inference time and conditioning the LLM's generation on that retrieved context, rather than relying solely on parametric memory.

This notebook implements a complete RAG pipeline — ingestion, chunking, embedding, indexing, retrieval, re-ranking, and generation — end to end, and wraps it in a browser-based interface so that a non-technical end user can:

1. Upload one or more documents (PDF, DOCX, or Excel spreadsheets), regardless of length;
2. Choose which language model answers their questions — a hosted proprietary model or a free open-weight model;
3. Ask questions of that document set using **typed text or spoken voice input**;
4. Receive answers grounded in the source material, with conversational memory across turns.

From a systems-design standpoint, the notable property of this implementation is that **every computationally expensive component in the pipeline — speech recognition, embedding generation, and cross-encoder re-ranking — runs on open-weight models with zero per-call cost**, and the only components that require an external network call are the final generation step (which itself offers a free open-weight option) and, optionally, a proprietary model for users who prefer it.

---

## 2. Business Rationale

### 2.1 Infrastructure Cost Reduction

A conventional voice-enabled document Q&A system is typically assembled from several paid, metered services:

| Pipeline Stage | Common Commercial Choice | Typical Cost Driver |
|---|---|---|
| Speech-to-Text | ElevenLabs, Deepgram, AssemblyAI, OpenAI Whisper API | Billed per audio-minute processed |
| Embeddings | OpenAI `text-embedding-3`, Cohere Embed | Billed per token embedded |
| Re-ranking | Cohere Rerank, proprietary reranking APIs | Billed per query/document pair scored |
| Generation | GPT-4-class or Gemini-class hosted models | Billed per input/output token |

Each of these is a recurring, usage-scaled operating expense. At even moderate query volumes (e.g., a call center or internal knowledge-base assistant fielding thousands of queries per day), these costs compound quickly and scale linearly — or worse — with adoption.

This implementation demonstrates that **comparable functional capability can be achieved by substituting open-weight models at three of the four pipeline stages**:

- **Speech-to-Text**: `openai/whisper-base` (Whisper, open-weight, MIT-licensed) runs locally within the compute session, transcribing microphone input with no per-minute billing and no dependency on a commercial voice API such as ElevenLabs. For an organization evaluating voice-interface costs at scale, this alone can represent a substantial recurring line-item eliminated.
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`, an open-weight sentence embedding model, runs locally with no per-token billing, no external API latency, and no data leaving the local environment during the embedding step.
- **Re-ranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2`, an open-weight cross-encoder, performs semantic re-ranking of candidate passages entirely locally, avoiding the per-query cost structure of commercial re-ranking APIs.
- **Generation**: the pipeline supports a fully open-weight model (Qwen2.5-7B-Instruct, Apache 2.0 license) accessed via the free tier of the Hugging Face Inference API, so an organization or individual without budget for a proprietary LLM subscription can still operate the complete system end-to-end at no direct cost, while retaining the option to switch to a proprietary model when higher output quality is warranted for a given use case.

The practical implication is that the marginal cost of operating this system approaches the cost of compute alone (or zero additional cost, when run on freely available infrastructure such as Google Colab), rather than scaling with a metered, per-call commercial pricing model. This is particularly material for early-stage proof-of-concept work, academic research, and cost-sensitive deployments where the value case for a fully proprietary stack has not yet been established.

### 2.2 Vendor Flexibility and Risk Mitigation

By architecting the generation layer as **provider-agnostic** rather than hard-coding a single LLM vendor, the system avoids single-vendor lock-in and provides organizational resilience against API deprecations, pricing changes, rate limits, and regional access restrictions. A user blocked from provisioning a Google API key (e.g., due to corporate network policy, account restrictions, or geographic availability) is not excluded from using the system — they can select the open-weight pathway instead, with no code changes required.

### 2.3 Applicability Beyond the Demonstration

While this notebook is scoped as a document Q&A assistant, the underlying architecture — ingestion → chunking → embedding → indexing → retrieval → re-ranking → grounded generation → conversational memory — is the same architecture that underlies production-grade:

- **Website chatbots** that answer visitor questions from a company's own documentation, FAQs, or product manuals;
- **Call center virtual agents** that retrieve policy or procedural information in real time during a live interaction, with the voice pipeline demonstrated here forming the basis of a speech-in/speech-out agent;
- **Internal knowledge-mining tools** that allow analysts to query large volumes of structured (spreadsheet) and unstructured (PDF/DOCX) organizational data using natural language, surfacing insights without manual document review.

Extending this notebook to any of these production contexts is primarily a matter of adding orchestration, authentication, and channel-integration layers (e.g., a telephony interface for a call center, or a website embed for a chatbot) **on top of** the retrieval-and-generation core demonstrated here — the core reasoning pipeline does not need to be re-architected.

---

## 3. Technical Architecture

### 3.1 Pipeline Stages

```
 Document Upload (PDF / DOCX / XLSX)
            │
            ▼
   Text Extraction (pypdf / python-docx / pandas)
            │
            ▼
   Recursive Character Chunking (LangChain RecursiveCharacterTextSplitter)
            │
            ▼
   Dense Embedding (sentence-transformers/all-MiniLM-L6-v2)
            │
            ▼
   Vector Indexing (FAISS)
            │
            ▼
   User Query (typed text OR transcribed speech via Whisper)
            │
            ▼
   Broad Retrieval — top-k=20 nearest neighbors (FAISS similarity search)
            │
            ▼
   Cross-Encoder Re-ranking — top-n=7 (cross-encoder/ms-marco-MiniLM-L-6-v2)
            │
            ▼
   Prompt Construction (retrieved context + rolling conversational memory)
            │
            ▼
   Generation — branches to one of two providers:
       ├── Gemini 2.5 Flash (Google Generative AI API)
       └── Qwen2.5-7B-Instruct (Hugging Face Inference API, open-weight)
            │
            ▼
   Grounded, Source-Constrained Answer
```

### 3.2 Document Ingestion and Chunking

The system accepts **PDF**, **DOCX**, and **XLSX/XLS** files of arbitrary length through a unified upload interface. Text is extracted using format-specific parsers (`pypdf.PdfReader` for PDF, `python-docx.Document` for Word, `pandas.read_excel` for spreadsheets) and concatenated into a single working corpus per session.

Rather than embedding entire documents as monolithic units — which exceeds most embedding models' effective context window and dilutes semantic specificity — the corpus is segmented using LangChain's `RecursiveCharacterTextSplitter` with a **chunk size of 700 characters and a 150-character overlap**. Recursive chunking attempts to split along natural linguistic boundaries (paragraphs, then sentences, then words) before falling back to hard character limits, preserving local coherence within each chunk. The 150-character overlap mitigates the risk of severing a relevant fact precisely at a chunk boundary, at a modest cost in index redundancy — a standard and well-established trade-off in production RAG systems.

This design allows large PDFs (e.g., multi-hundred-page reports) and large spreadsheets to be reduced to retrievable, semantically coherent units that a downstream LLM can reason over without requiring the full document in context.

### 3.3 Retrieval and Two-Stage Re-ranking

The system employs a **two-stage retrieval architecture**, a well-documented pattern for improving retrieval precision beyond what a single-stage dense retriever achieves in isolation:

1. **Stage 1 — Broad Recall (Bi-Encoder / FAISS):** The user's query is embedded using the same MiniLM embedding model used for indexing, and FAISS performs an approximate nearest-neighbor search to retrieve the top **k = 20** candidate chunks by cosine/L2 similarity. Bi-encoder retrieval is computationally cheap and scales well, but because the query and document are embedded independently, it can miss fine-grained semantic relationships between them.

2. **Stage 2 — Precision Re-ranking (Cross-Encoder):** The 20 candidates are then jointly re-scored by a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`), which processes the query and each candidate passage **together** in a single forward pass. This joint encoding allows the model to capture query-document interaction effects that bi-encoder similarity cannot, at the cost of being too computationally expensive to run over an entire corpus — which is precisely why it is applied only to the 20 pre-filtered candidates rather than the full index. The top **n = 7** re-ranked chunks are retained and passed into the generation prompt.

This retrieve-then-re-rank pattern is a standard technique in modern information retrieval and search systems for balancing recall (via the bi-encoder) against precision (via the cross-encoder) without incurring the computational cost of cross-encoding the entire corpus per query.

### 3.4 Provider-Agnostic Generation Layer (Branching Architecture)

The generation stage is implemented as a **runtime branch on a session-level provider flag** (`state.provider`), selected by the user in the application's Setup tab rather than hard-coded at build time:

```python
if state.provider == "gemini":
    response = state.gemini_model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=2048,
            temperature=0.3
        )
    )
    answer = response.text

elif state.provider == "hf_oss":
    answer = generate_with_open_model(prompt)
```

Both branches consume an **identical retrieved context and identical prompt structure** — the only variable is which model performs the final generation step. This is a deliberate architectural choice: it isolates model *choice* from pipeline *logic*, meaning retrieval quality, chunking strategy, and re-ranking are held constant regardless of which LLM answers the question, which is important for any downstream evaluation of generation quality attributable specifically to model choice rather than confounded pipeline variance.

Both providers are accessed as **remote API calls** with no local model hosting for generation:

- **Gemini 2.5 Flash** is called via Google's `google-generativeai` SDK, authenticated with a user-supplied Google API key, with **`max_output_tokens` capped at 2048** and **`temperature` set to 0.3** — a comparatively low sampling temperature chosen to favor factual, deterministic, context-grounded responses over creative variance, consistent with the retrieval-grounded objective of the system. A 2048-token output ceiling is sized to accommodate detailed, multi-paragraph answers with supporting explanation while bounding response latency and, where applicable, cost.
- **Qwen2.5-7B-Instruct** is called via the Hugging Face `InferenceClient.chat_completion()` interface, routed through the Hugging Face Inference Providers network (the `together` provider backend), authenticated with a user-supplied, free Hugging Face access token. No model weights are downloaded to, or executed within, the local runtime — this is a genuine remote inference call, architecturally equivalent to the Gemini call, using an ungated, Apache 2.0–licensed open-weight model.

Each provider's credential-configuration function performs a lightweight validation call before being accepted (a minimal "ping" completion), so that invalid or misconfigured credentials are surfaced immediately in the application UI rather than failing silently on the first real query.

### 3.5 Hallucination Mitigation via Retrieval Grounding

The system prompt explicitly constrains the model to answer **from the retrieved context only**, and to state plainly when the answer is not present in that context, rather than generating an unsupported response:

> *"You are a professional RAG assistant. Answer the question based on the provided context and history. If the answer isn't in the context, state that clearly."*

This is the core mechanism by which RAG architectures reduce hallucination relative to a bare LLM: rather than relying on the model's parametric memory (which may be outdated, incomplete, or simply incorrect for domain-specific content), the model is conditioned on retrieved, source-attributable passages at generation time. Because the retrieved chunks originate directly from the user's uploaded documents, an end user retains the ability to trace a generated answer back to its supporting source material, which is not possible with an ungrounded LLM response.

### 3.6 Conversational Memory

A rolling conversational memory of the **four most recent question–answer turns** is maintained per session and injected into each subsequent prompt, allowing the system to resolve follow-up questions, pronoun references, and topic continuity across a multi-turn dialogue without requiring the user to restate context — while bounding prompt length growth, since unbounded history accumulation would otherwise degrade both latency and cost as a conversation lengthens.

### 3.7 Speech Interface

Voice input is transcribed locally using `openai/whisper-base`, an open-weight automatic speech recognition (ASR) model, executed via the Hugging Face `transformers` pipeline API with automatic GPU/CPU device selection. Recorded audio is transcribed to text, populated into the chat input field, and processed through the identical text pipeline described above — meaning voice and typed input are functionally interchangeable entry points into the same RAG logic, with no separate code path or reduced capability for voice queries.

### 3.8 Application Layer

The complete system is exposed through [**Gradio**](https://www.gradio.app/), an open-source Python framework for rapidly building and sharing machine learning web interfaces. Gradio was selected specifically for its capacity to **eliminate frontend development overhead**: a fully interactive, shareable web application — with file upload, credential input, tabbed navigation, chat history, and live microphone capture — is defined declaratively in Python within the same notebook as the underlying pipeline, with no separate HTML/CSS/JavaScript build required. This makes the system suitable for rapid prototyping, stakeholder demonstration, and academic reproducibility, while still being representative of the interaction patterns (chat, file upload, multi-turn memory) required in a production deployment.

---

## 4. What This Notebook Actually Does (Plain-Language Summary)

For a reader less interested in the implementation detail and more interested in the practical capability:

- **Feed it a large PDF, Word document, or Excel spreadsheet.** The system breaks it into manageable, searchable pieces automatically — a user does not need to know or care how large the original document was.
- **Ask it questions in plain English, by typing or by speaking.** No need to know keywords, filenames, or where in the document the answer lives.
- **It answers using only what's actually in your documents** — rather than guessing or making something up, it looks up the most relevant passages first and bases its answer on those, and will say so if it can't find an answer in the material provided.
- **You choose the "brain" behind the answers**: a fast, high-quality commercial model (Google Gemini) if you have a Google account, or a completely free, open-source model if you don't — the rest of the system behaves identically either way.
- **It remembers the last few things you asked**, so you can have a natural back-and-forth conversation instead of repeating context every time.
- **It costs little to nothing to run**, because every component except the final answer-generation step runs on free, open-source models rather than paid subscriptions — and even that last step has a free option.

---

## 5. Extending This Architecture

Because the retrieval-and-generation core is decoupled from any specific interface, this notebook is a suitable starting point — rather than an end state — for:

- **Website / product chatbots**: replace the Gradio chat interface with a website-embedded chat widget that calls the same `rag_chat_v2()` function, pointed at a knowledge base of product documentation or support articles.
- **Call center virtual agents**: pair the existing Whisper-based speech-to-text pipeline with a text-to-speech layer and telephony integration (e.g., Twilio) to produce a fully voice-in/voice-out agent grounded in internal policy and procedure documents.
- **Business intelligence / data-mining assistants**: because the ingestion layer already parses Excel spreadsheets, this same pipeline can be pointed at structured operational data (financial reports, sales logs, inventory sheets) to allow non-technical stakeholders to query tabular business data conversationally rather than through manual spreadsheet review.
- **Multi-tenant, persistent knowledge bases**: the current implementation holds session state in memory (appropriate for a single-user demo); a production deployment would persist the FAISS index and swap the in-memory `RAGState` object for per-user or per-tenant session state.

---

## 6. Requirements and Setup

This notebook is designed to run in **Google Colab** with no local installation required. All dependencies are installed in the first cell.

**Credentials required (choose one):**
- A free Google API key ([Google AI Studio](https://aistudio.google.com/apikey)), **or**
- A free Hugging Face access token ([huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)) — a "Read" token is sufficient.

No API key, token, or document is hardcoded anywhere in this notebook. Each user supplies their own credentials at runtime through the application's Setup tab; credentials are held only in memory for the active session and are never written to disk or logged.

**To run:**
1. Open the notebook in Google Colab.
2. Run all cells in order (`Runtime → Run all`).
3. In the launched Gradio application, select an LLM provider, supply the corresponding credential, upload one or more documents, and click **Initialize RAG System**.
4. Switch to the **Chat & Voice** tab and begin querying your documents by text or speech.

---

## 7. Core Technologies

| Component | Model / Library | License / Access |
|---|---|---|
| Speech-to-Text | `openai/whisper-base` | Open-weight (MIT), local inference |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Open-weight, local inference |
| Vector Index | FAISS (Facebook AI Similarity Search) | Open-source, local |
| Chunking | LangChain `RecursiveCharacterTextSplitter` | Open-source |
| Re-ranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Open-weight, local inference |
| Generation (Option A) | Gemini 2.5 Flash | Proprietary, remote API (user-supplied key) |
| Generation (Option B) | Qwen2.5-7B-Instruct | Open-weight (Apache 2.0), remote API via Hugging Face Inference Providers |
| Application Interface | Gradio | Open-source |

---

## 8. Citation / Attribution

This implementation draws on the foundational RAG methodology described in:

> Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* Advances in Neural Information Processing Systems (NeurIPS).

---

## 9. Disclosures and Limitations

- **Session-scoped state**: knowledge base contents, chat history, and provider configuration are held in an in-memory Python object for the duration of the runtime session and are not persisted between sessions.
- **Shared session with `share=True`**: the Gradio application is launched with a public shareable link; because application state is not partitioned per visitor, this configuration is appropriate for individual/demonstration use rather than concurrent multi-user production deployment without further modification.
- **Not a substitute for professional advice**: as with any LLM-based system, outputs should be independently verified before being relied upon for legal, medical, financial, or other high-stakes decisions.
