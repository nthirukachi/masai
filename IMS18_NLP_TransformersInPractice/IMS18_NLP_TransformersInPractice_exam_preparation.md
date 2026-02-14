# IMS18: NLP & Transformers in Practice - Exam Preparation

> 📚 **Complete Exam Prep** for: RNNs, LSTMs, Encoder-Decoder, Attention, Transformers, BERT, GPT, Pre-training, Fine-tuning, RLHF, Tokenization, RAG

---

## Section A: Multiple Choice Questions (MCQ) - 15 Questions

### MCQ 1
**Question:** What does the hidden state H_t in RNNs represent?

**Options:**
- A) The final output of the network
- B) The model's memory of previous timesteps
- C) The learning rate
- D) The input embedding

**✅ Correct Answer:** B

**📖 Explanation:** H_t carries information from previous timesteps, acting as the model's short-term memory.

**❌ Why Others Are Wrong:**
- A) Y_t is the output, not H_t
- C) Learning rate is a hyperparameter, not a state
- D) Input embedding is X_t, not H_t

---

### MCQ 2
**Question:** LSTM solves the vanishing gradient problem by introducing:

**Options:**
- A) More layers
- B) Larger learning rate
- C) Gate mechanisms (Forget, Input, Output)
- D) Dropout regularization

**✅ Correct Answer:** C

**📖 Explanation:** Gates control information flow, allowing gradients to flow through without vanishing.

**❌ Why Others Are Wrong:**
- A) More layers worsen gradient flow
- B) Larger LR causes divergence, not stability
- D) Dropout is regularization, not memory solution

---

### MCQ 3
**Question:** In the Attention mechanism, the Query (Q) represents:

**Options:**
- A) The information each token stores
- B) What the current token is searching for
- C) The final output
- D) The weight matrix

**✅ Correct Answer:** B

**📖 Explanation:** Query represents what the current token wants to find in other tokens (like a search query).

**❌ Why Others Are Wrong:**
- A) That's the Value (V)
- C) Output is computed from attention scores
- D) Weight matrices are separate from QKV

---

### MCQ 4
**Question:** Why is the attention score scaled by √d_k?

**Options:**
- A) To increase computation speed
- B) To prevent softmax saturation with large values
- C) To reduce memory usage
- D) To add non-linearity

**✅ Correct Answer:** B

**📖 Explanation:** Large dot products cause softmax to output extreme values (0 or 1), leading to vanishing gradients.

**❌ Why Others Are Wrong:**
- A) Scaling adds computation, doesn't speed up
- C) Memory usage is same
- D) Scaling is linear operation

---

### MCQ 5
**Question:** What is the main difference between BERT and GPT?

**Options:**
- A) BERT is faster than GPT
- B) BERT is bidirectional (encoder); GPT is autoregressive (decoder)
- C) GPT has more parameters
- D) BERT is for images; GPT is for text

**✅ Correct Answer:** B

**📖 Explanation:** BERT uses encoder (sees full context); GPT uses decoder (sees only left context).

**❌ Why Others Are Wrong:**
- A) Speed depends on implementation
- C) Parameter count varies by version
- D) Both are for text

---

### MCQ 6
**Question:** RLHF stands for:

**Options:**
- A) Recurrent Learning with Hidden Feedback
- B) Reinforcement Learning from Human Feedback
- C) Regularized Learning with High Fidelity
- D) Robust Learning for High Frequency

**✅ Correct Answer:** B

**📖 Explanation:** RLHF uses human ratings to train models to be helpful, harmless, and honest.

---

### MCQ 7
**Question:** Which tokenization approach breaks "unhappiness" into subwords?

**Options:**
- A) Character-level tokenization
- B) Word-level tokenization
- C) Subword tokenization (BPE/WordPiece)
- D) Sentence tokenization

**✅ Correct Answer:** C

**📖 Explanation:** BPE/WordPiece breaks words into smaller meaningful units: "un" + "happi" + "ness".

---

### MCQ 8
**Question:** In the Transformer architecture, positional encoding is needed because:

**Options:**
- A) The model needs to count tokens
- B) Parallel processing loses sequence order information
- C) It improves accuracy
- D) It reduces training time

**✅ Correct Answer:** B

**📖 Explanation:** Unlike RNN, Transformer processes all tokens simultaneously, losing position info.

---

### MCQ 9
**Question:** What is the purpose of the [CLS] token in BERT?

**Options:**
- A) Marks the end of sentence
- B) Separates two sentences
- C) Aggregates sequence-level information for classification
- D) Represents unknown words

**✅ Correct Answer:** C

**📖 Explanation:** [CLS] token's embedding is used for classification tasks as it summarizes the sequence.

---

### MCQ 10
**Question:** Pre-training on Common Crawl data means:

**Options:**
- A) Training on labeled data
- B) Training on massive unlabeled web data
- C) Training on medical records
- D) Training on image data

**✅ Correct Answer:** B

**📖 Explanation:** Common Crawl contains 364 TB of web pages - general, unlabeled text data.

---

### MCQ 11
**Question:** Multi-head attention uses multiple heads because:

**Options:**
- A) It's faster than single head
- B) Different heads learn different types of relationships
- C) It uses less memory
- D) It's easier to implement

**✅ Correct Answer:** B

**📖 Explanation:** Each head can focus on different aspects (syntax, semantics, positions, etc.).

---

### MCQ 12
**Question:** RAG (Retrieval Augmented Generation) solves which problem?

**Options:**
- A) Slow inference
- B) LLM's knowledge cutoff and lack of private data access
- C) High training cost
- D) Model size

**✅ Correct Answer:** B

**📖 Explanation:** RAG retrieves current/private documents before generation, overcoming LLM limitations.

---

### MCQ 13
**Question:** Static embeddings (Word2Vec) have which limitation?

**Options:**
- A) Too slow to compute
- B) Same vector for word regardless of context
- C) Too large vocabulary
- D) Cannot handle English

**✅ Correct Answer:** B

**📖 Explanation:** "bank" (money) and "bank" (river) get same vector in Word2Vec.

---

### MCQ 14
**Question:** The formula for Scaled Dot-Product Attention is:

**Options:**
- A) softmax(QK / d_k) × V
- B) softmax(QK^T / √d_k) × V
- C) tanh(Q + K) × V
- D) sigmoid(QK) × V

**✅ Correct Answer:** B

**📖 Explanation:** Q multiplied by K transpose, scaled by square root of d_k, then softmax and multiply by V.

---

### MCQ 15
**Question:** Context length in LLMs refers to:

**Options:**
- A) Number of layers
- B) Maximum tokens the model can process at once
- C) Training time
- D) Number of parameters

**✅ Correct Answer:** B

**📖 Explanation:** Context length determines how much text the model can "see" - e.g., 128K tokens for GPT-4.

---

## Section B: Multiple Select Questions (MSQ) - 12 Questions

### MSQ 1
**Question:** Which are valid activation functions in LSTM? (Select ALL that apply)

**Options:**
- A) Sigmoid (for gates)
- B) TanH (for cell state)
- C) ReLU (for output)
- D) Softmax (for gates)

**✅ Correct Answers:** A, B

**📖 Explanation:** Sigmoid (0-1) for gate decisions; TanH (-1 to 1) for value normalization in cell state.

---

### MSQ 2
**Question:** Which tasks are transformers commonly used for? (Select ALL)

**Options:**
- A) Machine Translation
- B) Text Generation
- C) Image Classification
- D) Question Answering
- E) Sentiment Analysis

**✅ Correct Answers:** A, B, C, D, E

**📖 Explanation:** Transformers are used for all these - even images via Vision Transformers (ViT).

---

### MSQ 3
**Question:** Pre-training a large language model requires: (Select ALL)

**Options:**
- A) Massive compute (1000s of GPUs)
- B) Weeks to months of training
- C) Labeled data only
- D) Large unlabeled text corpus
- E) Millions of dollars

**✅ Correct Answers:** A, B, D, E

**📖 Explanation:** Pre-training uses unlabeled data and is extremely expensive.

**❌ Why C is wrong:** Pre-training uses self-supervised learning (predict next token), not labeled data.

---

### MSQ 4
**Question:** Which are special tokens used by BERT? (Select ALL)

**Options:**
- A) [CLS]
- B) [SEP]
- C) [MASK]
- D) [PAD]
- E) <bos>

**✅ Correct Answers:** A, B, C, D

**📖 Explanation:** <bos> is used by GPT, not BERT. BERT uses [CLS], [SEP], [MASK], [PAD].

---

### MSQ 5
**Question:** RLHF training involves: (Select ALL)

**Options:**
- A) Human annotators rating outputs
- B) Multiple output generation per input
- C) Training a reward model
- D) Reinforcement learning optimization
- E) Supervised learning only

**✅ Correct Answers:** A, B, C, D

**📖 Explanation:** RLHF combines all these except E - it's not just supervised learning.

---

### MSQ 6
**Question:** Benefits of Multi-Head Attention include: (Select ALL)

**Options:**
- A) Learning different relationship types
- B) Parallel computation
- C) Attending to different positions
- D) Reduced model size
- E) Better gradient flow

**✅ Correct Answers:** A, B, C

**📖 Explanation:** Multi-head doesn't reduce size (adds parameters) but enables diverse learning.

---

### MSQ 7
**Question:** Subword tokenization methods include: (Select ALL)

**Options:**
- A) Byte Pair Encoding (BPE)
- B) WordPiece
- C) SentencePiece
- D) Whitespace splitting
- E) Unigram

**✅ Correct Answers:** A, B, C, E

**📖 Explanation:** Whitespace splitting is simple word tokenization, not subword.

---

### MSQ 8
**Question:** BERT is best suited for: (Select ALL)

**Options:**
- A) Text classification
- B) Named Entity Recognition
- C) Question Answering (extractive)
- D) Open-ended text generation
- E) Sentence similarity

**✅ Correct Answers:** A, B, C, E

**📖 Explanation:** BERT (encoder) is for understanding tasks. GPT (decoder) is for generation (D).

---

### MSQ 9
**Question:** Which affect context length requirements? (Select ALL)

**Options:**
- A) Document length
- B) Conversation history
- C) Model size
- D) Task complexity
- E) Available memory

**✅ Correct Answers:** A, B, D, E

**📖 Explanation:** Model size is separate from context length (they're different dimensions).

---

### MSQ 10
**Question:** Encoder-Decoder architecture can be used for: (Select ALL)

**Options:**
- A) Machine Translation
- B) Text Summarization
- C) Image Classification
- D) Question Answering
- E) Chatbots

**✅ Correct Answers:** A, B, D, E

**📖 Explanation:** Image classification typically uses encoder-only (CNN or ViT).

---

### MSQ 11
**Question:** Components of Transformer encoder block: (Select ALL)

**Options:**
- A) Multi-Head Self-Attention
- B) Feed Forward Network
- C) Layer Normalization
- D) Residual Connections
- E) Convolutional layers

**✅ Correct Answers:** A, B, C, D

**📖 Explanation:** Transformers don't use convolutional layers (that's CNNs).

---

### MSQ 12
**Question:** Fine-tuning can be used to adapt models for: (Select ALL)

**Options:**
- A) Specific domain (medical, legal)
- B) Different language
- C) Different output format
- D) Specific task (coding, chat)
- E) Changing architecture

**✅ Correct Answers:** A, B, C, D

**📖 Explanation:** Architecture changes require retraining, not fine-tuning.

---

## Section C: Numerical Questions - 6 Questions

### Numerical 1
**Question:** If a transformer has 8 attention heads and model dimension d_model = 512, what is the dimension of each head (d_k)?

**Answer:** d_k = d_model / num_heads = 512 / 8 = **64**

---

### Numerical 2
**Question:** Calculate the attention score (before softmax) for Q = [1, 0], K = [0.5, 0.5], with d_k = 2.

**Solution:**
- QK^T = (1×0.5) + (0×0.5) = 0.5
- Scaled: 0.5 / √2 = 0.5 / 1.414 = **0.354**

---

### Numerical 3
**Question:** A model has context length of 4096 tokens. If 1 token ≈ 4 characters, approximately how many characters can it process?

**Answer:** 4096 × 4 = **16,384 characters** (≈ 2,700 words)

---

### Numerical 4
**Question:** BERT-base has 12 layers, 12 attention heads, hidden size 768. Calculate total attention heads across all layers.

**Answer:** 12 layers × 12 heads = **144 attention heads**

---

### Numerical 5
**Question:** If vocabulary size is 50,000 and embedding dimension is 768, what is the embedding matrix size?

**Answer:** 50,000 × 768 = **38,400,000 parameters** (38.4M)

---

### Numerical 6
**Question:** In RLHF, human annotator gives scores: Output1 = 0.75, Output2 = 0.20, Output3 = 0.05. Which output will model prefer?

**Answer:** **Output1** (highest score 0.75 indicates best quality)

---

## Section D: Fill-in-the-Blanks - 8 Questions

1. The paper "_________ is All You Need" introduced the Transformer architecture in 2017.
   - **Answer:** Attention

2. LSTM has three gates: Forget gate, Input gate, and _________ gate.
   - **Answer:** Output

3. BERT stands for Bidirectional _________ Representations from Transformers.
   - **Answer:** Encoder

4. GPT stands for Generative _________ Transformer.
   - **Answer:** Pre-trained

5. The cell state C_t in LSTM represents _________ memory.
   - **Answer:** Long-term

6. Common Crawl dataset is approximately _________ terabytes of web data.
   - **Answer:** 364

7. In attention formula, softmax(QK^T / √d_k) × V, the division by √d_k is called _________.
   - **Answer:** Scaling

8. _________ embeddings give same vector for a word regardless of context.
   - **Answer:** Static

---

## Section E: Quick Revision Points

### NLP Evolution Timeline
```
1990s: RNN (short memory)
       ↓
Mid-1990s: LSTM (long+short memory)
       ↓
2014: Seq2Seq (encoder-decoder)
       ↓
2017: Transformer (attention only)
       ↓
2018: BERT (encoder), GPT (decoder)
       ↓
2022+: ChatGPT (RLHF aligned)
```

### Memory Types in LSTM
| Memory | Symbol | Duration | Gate |
|--------|--------|----------|------|
| Long-term | C_t | Across many steps | Forget + Input |
| Short-term | H_t | Recent context | Output |

### BERT vs GPT Comparison
| Feature | BERT | GPT |
|---------|------|-----|
| Architecture | Encoder-only | Decoder-only |
| Direction | Bidirectional | Left-to-right |
| Training | Masked LM | Next token prediction |
| Best for | Understanding | Generation |

### Tokenization Quick Reference
| Method | Used By | Approach |
|--------|---------|----------|
| BPE | GPT | Merge frequent pairs |
| WordPiece | BERT | Score-based merging |
| SentencePiece | T5-LLaMA | Language-independent |

---

## Section F: Shortcuts & Cheat Codes

### Quick Formulas
- **Attention:** softmax(QK^T / √d_k) × V
- **Token estimate:** 1 token ≈ 4 characters ≈ 0.75 words
- **Head dimension:** d_k = d_model / num_heads

### Interview One-Liners
- **RNN:** Sequential processing with hidden state feedback
- **LSTM:** Gates control long-term and short-term memory
- **Attention:** Weighted importance of all tokens
- **Transformer:** Parallel attention without recurrence
- **BERT:** Bidirectional understanding
- **GPT:** Autoregressive generation
- **RLHF:** Human ratings for alignment

### Common Mistakes to Avoid
1. Don't confuse H_t (hidden) with C_t (cell)
2. BERT for understanding, GPT for generation
3. Scaling prevents vanishing gradients
4. Static embeddings don't handle polysemy
5. Pre-training is unsupervised, fine-tuning can be supervised

---

**📚 End of Exam Preparation. Good luck!** 🎓
