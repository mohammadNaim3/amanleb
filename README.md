# 🇱🇧 AmanLeb

**AI-Powered Scam & Smishing Detection for Lebanon**

AmanLeb is a multilingual SMS safety prototype designed to detect **ham**, **spam**, and **smishing** messages and provide trusted, source-grounded safety guidance for suspicious cases.

The project combines:

- exploratory data analysis and data cleaning,
- a classical **TF-IDF + Logistic Regression** baseline,
- a fine-tuned **multilingual DistilBERT** classifier,
- confidence-based safety routing,
- multilingual semantic retrieval using **Sentence Transformers + ChromaDB**,
- deterministic extractive guidance from trusted Lebanese cybersecurity sources,
- and a **Streamlit** user interface.

---

## 1. Problem

SMS scams and phishing messages often imitate banks, payment services, telecom companies, or other trusted organizations. In Lebanon, these messages may also appear in multiple languages or mixed-language forms.

AmanLeb was built as a prototype to:

1. classify an SMS as **ham**, **spam**, or **smishing**,
2. expose the model's class probabilities,
3. apply a conservative safety-routing layer,
4. retrieve relevant guidance from trusted Lebanese sources when additional review is needed,
5. provide source-grounded explanations and recommended actions without allowing retrieval to overwrite the classifier's prediction.

---

## 2. System Architecture

```text
User SMS
   │
   ▼
Streamlit Interface
   │
   ▼
Weighted Multilingual DistilBERT
   │
   ├── HAM
   ├── SPAM
   └── SMISHING
   │
   ▼
Confidence-Based Safety Routing
   │
   ├── High-confidence HAM ─────► Likely Safe
   │
   ├── Low-confidence HAM ──────► Needs Review
   │
   └── SPAM / SMISHING ─────────► Suspicious
                                    │
                                    ▼
                         Multilingual Semantic Retrieval
                                    │
                                    ▼
                           Trusted Lebanese Sources
                                    │
                                    ▼
                        Deterministic Extractive Guidance
```

### Important design rule

The RAG component **does not reclassify the SMS**.

The Transformer produces the model prediction. The safety-routing layer decides whether additional evidence should be retrieved. RAG only provides supporting evidence and safety guidance.

---

## 3. Dataset

The project uses the Mendeley dataset:

**SMS PHISHING DATASET FOR MACHINE LEARNING AND PATTERN RECOGNITION**

Original dataset size:

- **5,971 rows**

Original columns:

- `LABEL`
- `TEXT`
- `URL`
- `EMAIL`
- `PHONE`

### Label normalization

The original labels contained inconsistent capitalization:

```text
ham
Smishing
smishing
spam
Spam
```

They were normalized to:

```text
ham
spam
smishing
```

### Data cleaning

The cleaning process included:

- label normalization,
- binary metadata normalization,
- duplicate-message inspection,
- conflicting-label detection,
- removal of conflicting duplicate texts,
- removal of remaining duplicate texts,
- missing-value checks.

Final cleaned dataset:

- **5,947 rows**
- **5,947 unique SMS texts**

### Class distribution

| Class | Approx. Share |
|---|---:|
| Ham | 81.28% |
| Smishing | 10.53% |
| Spam | 8.19% |

Because the dataset is strongly imbalanced, evaluation focuses on more than accuracy alone.

Important metrics include:

- precision,
- recall,
- F1-score,
- macro F1,
- confusion matrices,
- suspicious messages incorrectly classified as ham.

---

## 4. Exploratory Data Analysis

EDA showed that:

- ham messages are generally shorter on average,
- suspicious messages are often longer,
- message lengths still overlap substantially between classes,
- URL, email, and phone indicators can be useful but are noisy,
- the supplied metadata columns should not be treated as perfect phishing indicators.

For the classical baseline, the project therefore uses **message text only** rather than relying on potentially noisy metadata.

---

## 5. Classical Baseline — TF-IDF + Logistic Regression

The classical model uses:

```text
SMS Text
   │
   ▼
TF-IDF Vectorization
   │
   ▼
Logistic Regression
```

The same stratified split is used throughout the project:

- **70% training**
- **15% validation**
- **15% test**
- `random_state=42`

Split sizes:

| Split | Messages |
|---|---:|
| Train | 4,162 |
| Validation | 892 |
| Test | 893 |

A standard Logistic Regression model was first evaluated. Because of the class imbalance, a second model using:

```python
class_weight="balanced"
```

was tested and selected using validation performance.

### Final Logistic Regression test results

| Metric | Result |
|---|---:|
| Accuracy | **96.08%** |
| Macro F1 | **0.884** |
| Suspicious → Ham errors | **5** |
| Smishing → Ham errors | **0** |

Final confusion matrix:

```text
                Predicted
              Ham  Spam  Smishing

Actual Ham    717    4      5
Actual Spam     5   61      7
Actual Smish    0   14     80
```

This baseline provides a strong and interpretable reference point for the Transformer model.

---

## 6. Multilingual Transformer Classifier

The final classifier is based on:

**`distilbert/distilbert-base-multilingual-cased`**

The project fine-tunes the full model for three-class sequence classification:

```text
0 → ham
1 → spam
2 → smishing
```

### Tokenization

A multilingual WordPiece tokenizer is used.

A token-length audit showed that only a very small number of training messages exceeded 128 tokens, so the project uses:

```python
max_length = 128
```

Dynamic padding is used during training.

### Transformer experiments

Two Transformer experiments were performed:

1. unweighted multilingual DistilBERT,
2. class-weighted multilingual DistilBERT.

The weighted model uses training-set class weights inside the cross-entropy loss.

The best model was selected using **validation macro F1**, not the test set.

### Final Transformer test results

| Metric | Result |
|---|---:|
| Accuracy | **97.3%** |
| Macro F1 | **0.905** |
| Suspicious → Ham errors | **2** |
| Smishing → Ham errors | **0** |

Per-class test performance:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Ham | 0.997 | 1.000 | 0.999 |
| Spam | 0.836 | 0.836 | 0.836 |
| Smishing | 0.891 | 0.872 | 0.882 |

Final confusion matrix:

```text
                Predicted
              Ham  Spam  Smishing

Actual Ham    726    0      0
Actual Spam     2   61     10
Actual Smish    0   12     82
```

The weighted multilingual DistilBERT model was selected as AmanLeb's final classifier.

> The zero smishing-to-ham count refers only to the held-out test split used in this project. It should not be interpreted as a guarantee that the model can never miss a smishing message.

---

## 7. Safety Routing

AmanLeb separates the **model prediction** from the **user-facing safety status**.

The routing logic is:

```text
Predicted HAM + HAM probability ≥ 0.80
        │
        └── Likely Safe → RAG is skipped

Predicted HAM + HAM probability < 0.80
        │
        └── Needs Review → RAG runs

Predicted SPAM or SMISHING
        │
        └── Suspicious → RAG runs
```

The **0.80 HAM threshold is a conservative prototype heuristic**.

It was not statistically optimized and should not be interpreted as a calibrated production safety threshold.

This routing was added after observing that a synthetic scam-like message could occasionally receive a ham prediction with moderate confidence. The extra review path allows AmanLeb to remain cautious when the classifier is uncertain.

---

## 8. Retrieval-Augmented Safety Guidance

AmanLeb uses multilingual semantic retrieval to search trusted Lebanese cybersecurity guidance.

### Embedding model

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

The embedding model produces **384-dimensional multilingual embeddings**.

### Vector database

The project uses:

**ChromaDB**

Source documents are:

1. scraped,
2. cleaned,
3. split into overlapping word chunks,
4. embedded,
5. stored with source metadata.

Chunking configuration:

```text
chunk size: 80 words
overlap:    20 words
```

Metadata includes:

- source ID,
- title,
- organization,
- URL,
- chunk index.

This allows retrieved evidence to remain traceable to the original source.

---

## 9. Trusted Lebanese Sources

The prototype currently uses three sources.

### Lebanese Internal Security Forces — Fraudulent SMS Warning

**Organization:** Lebanese Internal Security Forces

Focuses on fraudulent SMS messages impersonating a money-transfer company and warns users about attempts to steal personal, financial, and banking information.

Source:

https://isf.gov.lb/news/fraudulent-text-messages-impersonating-a-money-transfer-company-beware-of-the-theft-of-your-personal-data-and-money/

### Lebanese Internal Security Forces — Internet Security Awareness

**Organization:** Lebanese Internal Security Forces

Provides broader internet and account-security guidance.

Source:

https://isf.gov.lb/internet-security-awareness/

### Alfa Security Tips

**Organization:** Alfa

Provides general account and security recommendations.

Source:

https://www.alfa.com.lb/en/support/security-tips

---

## 10. Safe Extractive Guidance

The final deployed application does not allow a language model to freely invent safety advice.

Instead, AmanLeb:

1. retrieves the most relevant trusted-source chunk,
2. adds the next chunk from the same source when available,
3. searches the retrieved evidence for a sufficiently specific fraud-related sentence,
4. extracts recommended actions directly from the trusted evidence,
5. returns the official source and URL.

Typical extracted actions include guidance such as:

- not clicking links contained in suspicious messages,
- not entering personal or financial information,
- not forwarding suspicious messages.

If sufficiently specific evidence is not found, AmanLeb says so rather than fabricating an explanation.

---

## 11. Qwen Grounded-Generation Experiment

During development, AmanLeb also evaluated:

**`Qwen/Qwen2.5-1.5B-Instruct`**

The goal was to test whether a small instruction-tuned language model could convert retrieved evidence into a concise user-facing explanation.

The experiment showed that the model could produce fluent text but was not consistently reliable under strict grounding requirements. It sometimes:

- introduced unsupported claims,
- ignored formatting requirements,
- failed evidence-citation constraints.

A deterministic validator was added to detect these failures.

### Final engineering decision

Qwen is **not required by the final Streamlit runtime**.

The deployed prototype instead prioritizes:

```text
Classifier
+ Safety Routing
+ Multilingual Retrieval
+ Deterministic Extractive Guidance
```

This reduces local resource requirements and prioritizes traceability over generative fluency.

Generative response synthesis remains an experimental/future component.

---

## 12. Challenge Tests

After source-code cleanup, the final application was retested on six representative messages and reproduced the same behavior.

| Test | Model / Safety Outcome | Evaluation |
|---|---|---|
| Normal English message | HAM 99.6% — Likely Safe | ✅ Strong pass |
| Promotional prize message | SMISHING 88.0% — Suspicious | ✅ suspicious detection, ⚠️ exact spam/smishing confusion |
| English banking scam | SMISHING 98.7% — Suspicious | ✅ Strong pass |
| Arabic banking scam | SMISHING 98.0% — Suspicious | ✅ Strong multilingual pass |
| French banking scam | SMISHING 98.4% — Suspicious | ✅ Strong multilingual pass |
| English + Arabizi scam | SMISHING 98.7% — Suspicious | ✅ Strong mixed-language pass |

The tests demonstrate strong suspicious-message detection across English, Arabic, French, and mixed-language inputs.

They also exposed an important RAG limitation: the retriever can find a relevant trusted source while the strict extractive layer may still fail to find a sufficiently specific sentence or action inside the selected evidence.

AmanLeb intentionally prefers returning no specific explanation over generating unsupported advice.

---

## 13. Project Structure

```text
amanleb/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   │   └── Dataset_5971.csv
│   │
│   └── processed/
│       └── Dataset_5971_clean.csv
│
├── models/
│   ├── amanleb_final_transformer/
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── tokenizer.json
│   │   └── tokenizer_config.json
│   │
│   └── chroma_db/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_logistic_regression.ipynb
│   ├── 03_transformer_classifier.ipynb
│   └── 04_rag.ipynb
│
├── results/
│   └── screenshots/
│
├── report/
│
├── src/
│   ├── __init__.py
│   └── rag.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 14. Installation

Clone the repository and enter the project folder:

```bash
git clone <YOUR_REPOSITORY_URL>
cd amanleb
```

Create a virtual environment.

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

---

## Model Files

The trained AmanLeb Transformer weights are hosted on Hugging Face:

**Hugging Face Model:**  
https://huggingface.co/mohammadNaim3/AmanLeb

Download the model files and place them at:

```text
models/amanleb_final_transformer/

## 16. Running AmanLeb

From the project root:

```bash
python -m streamlit run app/app.py
```

Then open the local Streamlit URL shown in the terminal.

The classifier loads when the application starts.

The RAG engine is loaded lazily only when the message is:

- **Needs Review**, or
- **Suspicious**.

### First RAG use

The RAG component uses the multilingual Sentence Transformer embedding model and a persistent Chroma database.

If an existing Chroma collection is available, AmanLeb reuses it.

If the database is empty, the trusted-source documents are downloaded, cleaned, chunked, embedded, and stored automatically. Internet access is therefore required when the knowledge base or embedding model must be built/downloaded for the first time.

---

## 17. Notebooks

The notebooks document the development process in order.

### `01_data_exploration.ipynb`

Covers:

- raw-data inspection,
- label normalization,
- duplicate/conflict handling,
- class imbalance,
- message-length analysis,
- metadata analysis,
- cleaned-dataset export.

### `02_logistic_regression.ipynb`

Covers:

- stratified splitting,
- TF-IDF,
- baseline Logistic Regression,
- validation analysis,
- class weighting,
- final held-out testing,
- coefficient-based interpretability.

### `03_transformer_classifier.ipynb`

Covers:

- multilingual tokenization,
- token-length analysis,
- Hugging Face dataset preparation,
- unweighted DistilBERT,
- weighted DistilBERT,
- validation-based model selection,
- final held-out testing,
- safety-oriented error analysis.

### `04_rag.ipynb`

Covers:

- trusted-source collection,
- web extraction,
- document cleaning,
- chunking,
- multilingual embeddings,
- Chroma retrieval,
- metadata-enriched retrieval,
- multilingual retrieval tests,
- grounded-generation experiments,
- deterministic validation,
- extractive fallback,
- classifier and safety-routing integration.

---

## 18. Current Limitations

AmanLeb is a research/internship prototype, not a production anti-fraud system.

Current limitations include:

- relatively small training dataset,
- imperfect distinction between `spam` and `smishing`,
- limited Lebanon-specific multilingual labeled data,
- only three trusted sources in the current RAG knowledge base,
- strict extraction can return no explanation even when retrieval is broadly relevant,
- the 0.80 safety threshold is heuristic rather than calibrated,
- challenge tests are useful qualitative checks but are not a replacement for a larger external benchmark,
- performance may differ on new scam styles, organizations, dialects, Arabizi spelling patterns, shortened URLs, or adversarial messages.

For sensitive financial or account-related messages, users should still verify the message through the organization's official channels.

---

## 19. Future Work

Possible extensions include:

- collect a larger Lebanon-specific multilingual SMS dataset,
- improve Arabizi and Lebanese-dialect coverage,
- expand the trusted-source cybersecurity knowledge base,
- improve top-k retrieval and evidence reranking,
- improve extractive evidence selection,
- statistically calibrate the safety-routing threshold,
- evaluate additional classical ML models,
- experiment with stronger grounded generation,
- add a full conversational chatbot/API layer,
- add dedicated agentic tool use,
- support automatic trusted-source knowledge-base refresh,
- perform larger-scale external and adversarial evaluation.

---

## 20. Final Model Comparison

| Model | Test Accuracy | Macro F1 | Suspicious → Ham | Smishing → Ham |
|---|---:|---:|---:|---:|
| Balanced Logistic Regression | 96.08% | 0.884 | 5 | 0 |
| Weighted Multilingual DistilBERT | **97.3%** | **0.905** | **2** | **0** |

The weighted multilingual DistilBERT model provides the best overall performance and is used by the final AmanLeb application.

---

## 21. Disclaimer

AmanLeb is an educational AI prototype developed for scam and smishing research.

Its predictions and retrieved guidance should be treated as decision-support information, not as a guarantee that a message is safe or malicious.

When a message involves banking credentials, payment information, personal data, account access, or urgent financial requests, verify it independently using the organization's official communication channels.
