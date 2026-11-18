# Agentic AI — Principal Engineer Study Guide

*A structured learning document covering context windows, agent architecture, cost economics, failure modes, design patterns, enterprise deployment, vendor landscape, and the theoretical foundations of LLMs.*

**Audience:** Principal Engineer level — technical depth with enterprise framing for banking/regulated environments.

---

## Table of Contents

1. [Core Mental Model](#1-core-mental-model)
2. [The Harness Concept](#2-the-harness-concept)
3. [How Agents Manage Context](#3-how-agents-manage-context)
4. [Internal LLM Calls — What the Harness Uses the Model For](#4-internal-llm-calls)
5. [Cost, Caching, and KV Mechanics](#5-cost-caching-and-kv-mechanics)
6. [Copilot's Pricing Model and Model Behavior](#6-copilots-pricing-model)
7. [Failure Modes](#7-failure-modes)
8. [Design Patterns](#8-design-patterns)
9. [Enterprise Concerns and MRM](#9-enterprise-concerns)
10. [Multi-Agent Systems](#10-multi-agent-systems)
11. [Capstone Synthesis](#11-capstone-synthesis)
12. [The Vendor Application Layer — 7-Layer Stack](#12-vendor-application-layer)
13. [Wells Fargo Context: Tachyon and Two Governance Tracks](#13-wells-fargo-context)
14. [Foundation Model Providers — Technical Depth](#14-foundation-model-providers)
15. [Provider Highlights (Light Survey)](#15-provider-highlights)
16. [Theory: Tokenization](#16-theory-tokenization)
17. [Theory: Attention Mechanics](#17-theory-attention)
18. [Theory: RLHF and RLAIF](#18-theory-rlhf-rlaif)
19. [Theory: FLOPs](#19-theory-flops)
20. [Theory: Sampling and Temperature](#20-theory-sampling)

---

## 1. Core Mental Model

- **LLM = stateless function.** Text in → text out. Zero memory between calls. Same input + temp=0 → same output.
- **Harness = stateful wrapper.** Client-side code (Claude Code, Copilot, Cursor) that assembles context, executes tools, manages state. Agent-ness lives here, not in the model.
- **Context = working memory.** Finite, expensive, fills up as agent acts. Effective context < nominal context (lost-in-the-middle).
- **KV cache = performance only.** Vendor-side stateful but semantically inert. Speeds up the prefix; never carries meaning.
- **Harness uses LLM for judgment.** Compaction, planning, routing = extra LLM calls (often invisible). One user request can trigger 30–80 model calls.

### The agent loop

```
1. Read user message
2. Assemble prompt (system + tools + history + new message)
3. Send to LLM API
4. Model returns either:
   (a) text response → display, wait for next message
   (b) tool_use block → harness executes locally, appends
       tool_result, GOES BACK TO STEP 3
5. Repeat 3-4 until model returns plain text
```

The model never executes anything itself. It only emits intentions to call tools. The harness does the execution.

### Statelessness in practice

- A conversation's history is replayed on every turn — the model re-reads from scratch.
- Turn 50 of a chat sends turns 1–49 as input again.
- This is why model-swapping mid-session in Copilot works: the "session" lives in the IDE, not the model.

---

## 2. The Harness Concept

**Definition:** An agent harness is the deterministic software layer that wraps a stateless LLM to produce stateful, goal-directed behavior.

### Harness responsibilities

- Context assembly (building the prompt each turn)
- Output parsing (extracting tool calls, final answers)
- Tool execution (actually running the actions)
- Control flow (loop, terminate, escalate)
- State management (messages, scratchpads, memory files)
- Resource governance (context limits, token budgets, tool quotas)

### Why "harness"?

Borrowed from *test harness* and literal horse harness. The LLM is the engine doing cognitive work. The harness translates that work into useful motion and keeps the model pointed in the right direction.

### Senior-engineer framings

- **Vendor lock-in lives in the harness, not the model.** Model choice is often fungible; harness design is the strategic decision.
- **Most "the AI is dumb" complaints are harness problems.** Bad prompt assembly, bad tool design, improper compaction — the model is usually fine.

---

## 3. How Agents Manage Context

Five strategies, usually combined in production:

### 1. Truncation (the dumb baseline)
Drop oldest messages. Catastrophic — loses the original task and key decisions.

### 2. Compaction / summarization (the workhorse)
Harness asks the model to summarize conversation into compact form. Preserves system prompt, original request, recent turns, plus generated summary. Loses exact text of old tool outputs. Risk: summary is lossy or wrong.

### 3. Retrieval / RAG inside the agent
Store information externally; let the agent fetch via tools (grep, read, semantic search). This is why Claude Code works on million-line codebases despite a 200K window — it's not loading the codebase, it's querying it. Principle: **just-in-time context**.

### 4. Sub-agents (orchestrator-worker)
Spawn a sub-agent in its own fresh context. Sub-agent runs, returns summary. Parent's context grows by 500 tokens instead of 80K. Sub-agents trade tokens asymmetrically — spend in sub-agent to save in main context.

### 5. Persistent memory (external state)
File system as memory. `CLAUDE.md` is the canonical example. Quality of memory files often matters more than model choice.

### Context hierarchy

```
Hot context         → system prompt, recent turns, current tool results
Summarized          → compacted history (lossy)
Retrievable         → code, docs, prior outputs via tools
Persistent          → memory files, project docs (survive sessions)
```

Good agents move information DOWN this hierarchy as it ages.

---

## 4. Internal LLM Calls

The harness makes separate API calls the user never sees, often with cheaper/faster models.

### What harnesses use the LLM for

- **Compaction / summarization** — "summarize this conversation, preserving key decisions"
- **Title generation** — auto-generated chat titles
- **Memory extraction** — background extraction of facts to remember
- **Routing / classification** — "is this coding or chat?"
- **Retrieval reranking** — LLM reranks search results for relevance
- **Sub-agent orchestration** — parent delegates, sub-agent executes, parent integrates
- **Plan generation** — explicit plan before execution

### What is pure code (no LLM)

- Counting tokens (tokenizer library)
- Deciding when to trigger compaction (threshold check)
- Executing tool calls
- Parsing structured model output
- Appending messages, reading files, running tests
- Permission checks, rate limits, sandboxing

**Architectural insight:** The decision *whether* to do something is often LLM-driven. The *doing* itself is almost always code-driven.

### Cost implication

A single user request to Claude Code might trigger 30–80 model calls — main loop turns, compaction, sub-agents, title generation, memory extraction. Naïve cost estimates undercount by 10–50x.

---

## 5. Cost, Caching, and KV Mechanics

### Three common cost model errors

1. **Input and output priced differently.** Output tokens cost 4–5× input tokens.
2. **Agents make many model calls per user request** (sub-agents, compaction, etc.).
3. **Conversation re-sends entire history** — turn 50 sends turns 1–49 again as input.

A 50-turn session where context grows to 100K tokens doesn't cost 50× one turn — total input billed is on the order of 2.5M tokens without caching.

### KV cache — what it caches

When a transformer processes a prompt, every token gets converted into **key (K)** and **value (V)** tensors per layer. These depend only on that token and prior tokens. Process the same prefix twice → same KV tensors.

**Prompt caching:** provider saves the KV tensors from the first call's prefix; subsequent calls skip the forward pass for that prefix.

### Pricing structure (rough)

- **Cache write** (first call, populating): ~1.25× normal input price
- **Cache read** (subsequent hits): ~0.1× normal input price (10x cheaper)
- **Cache miss**: full input price

### What's cacheable

- Requires EXACT prefix match
- System prompt and tool schemas cacheable (don't change)
- One character modification in prefix busts the entire cache
- Order matters: stable content first, dynamic content last
- Compaction busts the cache (rewrites conversation)

### Latency

Prompt caching speeds up the prefill phase. A 100K-token prompt without caching: 3–8 seconds prefill. With fully-cached prefix: ~200ms. For agents where developers wait, latency savings often justify caching alone.

### Agent cost equation

```
total_cost ≈ Σ (per turn)
  cached_prefix_tokens × cache_read_price
+ uncached_input_tokens × input_price        (growing tail)
+ output_tokens × output_price
+ Σ sub_agent_costs                           (recursive)
+ Σ internal_LLM_call_costs                   (compaction, titles)
```

---

## 6. Copilot's Pricing Model

GitHub Copilot prices per "premium request" (a unit GitHub defines), not per token. GitHub pays per token on the backend but doesn't pass variable cost through directly.

### Why viable for GitHub

- Aggregation across users (most requests are cheap)
- Aggressive caching at scale
- Model multipliers — expensive models consume more "premium requests" per user request

### Should you overload each request?

**No.** Trap:
- Effective context still degrades (lost-in-middle)
- Long requests trigger truncation/compaction
- Failure modes harder to recover from
- GitHub rate-limits heavy users eventually

The actually-good strategy under request-pricing is the same as under token-pricing: **focused requests, clear scoping, retrieve don't preload.**

### Model behavior affects usage

- Agentic models (Claude Sonnet/Opus) do more per turn but have higher multipliers
- Models that stop and ask ("want to continue?") charge per request each time
- Re-work costs requests — early checkpoints can save overall

### Model selection heuristics

| Task type | Model choice |
|-----------|-------------|
| Well-scoped, well-understood | Agentic model (high work-per-request) |
| Exploratory, ambiguous, high-stakes | Model that checks in more often |
| Trivial edits | Cheapest model |

---

## 7. Failure Modes

All seven trace back to attention mechanics over lossy/biased/contaminated context.

### 1. Context rot
Quality degrades as session grows. Noise dilutes signal in attention. Signal: "this agent was better 30 minutes ago." **Fix:** fresh session.

### 2. Summary drift
Agent asserts wrong facts after compaction. Information that didn't make it into the summary has no K vector representing it anymore. **Diagnostic:** ask to re-read the source. If answer flips, drift confirmed.

### 3. Lost-in-the-middle
Critical info from turn 3 violated at turn 38. Attention weights bias toward start/end. **Mitigation:** pin constraints at top (system) or bottom (recent message). Never trust the middle to carry load.

### 4. Tool-call thrashing
Loops of similar greps/reads with no progress. Flat attention distribution → unfocused output → more tool calls → more context noise. **Diagnostic:** "what would unblock you?" — usually missing human info.

### 5. Premature convergence
Agent locks onto first hypothesis, ignores contrary evidence. Model's own committed framing has created strong Ks in context that dominate attention. **Mitigation:** "list three other possible causes and rule them out."

### 6. Prompt injection via tool results
Content read via tools contains instructions model treats as from user. Attention has no architectural concept of trust — it's content-based lookup. **Defense:** least-privilege tool design, bounded blast radius.

### 7. Poisoned context
Bad assumption persists despite corrections. Original wrong assertion's Ks are still in context competing for attention. **Fix:** fresh session with corrected premise as input #1. Don't trust "you're right, sorry."

### Meta-pattern

Almost every failure mode is a consequence of context being lossy, biased, or contaminated, combined with the model having no clean way to reset or distinguish trustworthy from untrustworthy content.

**Framing:** *"Attention is content-addressable memory with no garbage collection. Everything you put in stays in, competing for weight, until you start fresh."*

---

## 8. Design Patterns

### 1. ReAct (base loop)
Thought → Action → Observation → repeat. Default in every agent. Greedy, can wander.

### 2. Plan-and-Execute
Generate explicit plan, then execute step by step. Plan stays in context as anchor. Best when problem is well-understood; worse for genuinely exploratory work.

### 3. Orchestrator-Worker (sub-agents)
Main agent delegates bounded sub-tasks to workers with isolated contexts. Each worker returns a summary. Killer use: parallelism + context isolation. Cost multiplies.

### 4. Reflection (generator-critic)
Model produces output → separate critique step → revises. Catches surface errors well; shared blind spots survive.

### 5. Evaluator-Optimizer
Iterate until score crosses threshold. Strongest with **deterministic evaluator** (tests pass, type-check clean, schema valid).

### 6. Router
Front-end classifier dispatches to specialist agent/tool set. Use only when task types are genuinely distinct.

### 7. Human-in-the-Loop
Pause for approval at irreversible/high-stakes actions. **Non-negotiable for regulated work.** Design WHERE checkpoints sit — too few = unacceptable risk, too many = rubber-stamp.

### Patterns combine

Claude Code is approximately: ReAct + Plan-and-Execute + Orchestrator-Worker + Reflection + Evaluator-Optimizer + Human-in-the-Loop, layered.

### Vendor evaluation questions

1. What's the base loop?
2. Is there explicit planning?
3. How do they handle work that exceeds one context window?
4. What's their quality-checking mechanism?
5. Where are human checkpoints?
6. How is model selection routed?

---

## 9. Enterprise Concerns

### Core regulatory framing

In banking regulatory language, **an LLM is a model**. Once it's a model, **SR 11-7 (Model Risk Management)** applies. Related: SR 23-4 (third-party risk), OCC Bulletin 2023-17, NIST AI RMF 1.0.

### What MRM requires

1. **Inventory** — every model documented
2. **Independent validation** — separate team tests before deployment
3. **Ongoing monitoring** — drift, regressions, quality, incidents
4. **Documentation** — model cards, limitations, escalation paths
5. **Tiering** — 3–5 tier scheme determines validation rigor

### Auditability

For agents, log at minimum:
- Every user input and system prompt version
- Every model call (name, version)
- Every tool call with full input/output
- Every output returned
- All sub-agent invocations and traces

**Tamper-evident storage** is increasingly required.

### Data governance

- **Prompt data leakage** — enterprise tier contracts specify no training on your data
- **Cross-border data flow** — regional deployment for residency
- **Output classification** — outputs inherit input classification
- **Vendor model updates** — require version pinning and re-validation

### Evaluation discipline

- Task definition (precise)
- Evaluation set (curated, representative)
- Metrics (accuracy, refusal, latency, cost, escalation, harmful-output)
- Adversarial set (injection attempts, edge cases)
- Regression suite (run on every change)
- Production sampling (continuous human review of samples)

**Framing:** *"We don't deploy code without tests. We shouldn't deploy AI without evals."*

### Risk taxonomy (8 categories)

1. **Accuracy** — wrong outputs
2. **Bias/fairness** — disparate impact
3. **Security** — injection, exfiltration, jailbreak
4. **Privacy** — data exposure
5. **Operational** — outage, latency
6. **Compliance** — violates regulations
7. **Reputational** — public embarrassment
8. **Third-party** — vendor failure

### Shadow IT

Individual engineers using personal-tier accounts with bank code/data. Don't police — make the sanctioned path as good as the shadow path.

---

## 10. Multi-Agent Systems

### Disambiguation

1. **Sub-agents (orchestrator-worker)** — works, in production, boring sense
2. **Multi-role agents** — one agent playing roles sequentially (pattern, not multi-agent)
3. **Specialized agents in fixed pipeline** — mostly software pipeline dressed up
4. **True multi-agent systems** — autonomous agents with own goals; almost nothing in production
5. **"Agent swarms" / "AI societies"** — usually marketing

### Where multi-agent wins

- Parallelism on independent subproblems
- Context isolation for cleanliness
- Specialization with different tools or models

### Where multi-agent loses

- Communication bandwidth (lossy text interface)
- Coordination failures
- Cost multiplication
- Debuggability (which agent failed when?)
- Convergence hard (ping-pong)

### Empirical finding

A well-prompted single agent with good tools usually beats multi-agent on the same task, until the task scales beyond one context window.

**Framing:** *"Multi-agent is useful for parallelism and context isolation, not emergent intelligence. Right question isn't 'how many agents?' but 'smallest count that fits within context limits while preserving auditability?'"*

---

## 11. Capstone Synthesis

### Unified mental model (7 points)

1. The LLM is a stateless function.
2. The harness is the stateful wrapper that makes the model behave like an agent.
3. Context is the agent's working memory and the scarce resource.
4. The harness uses the LLM for judgment and uses code for execution.
5. Patterns are the answers to "how do we decompose?"
6. Failure modes are predictable consequences of the architecture.
7. Enterprise deployment is a model-risk problem.

### Talk tracks

**Tech-to-Tech:** *"The interesting thing about agents isn't the model — it's the harness. Model is fungible; harness defines behavior. We should design the harness layer most carefully."*

**Tech-to-Business:** *"AI agents can do real work and can fail in ways traditional software doesn't. The discipline is designing where they act autonomously vs. where humans approve, measuring outputs continuously, and treating them as models under MRM."*

### Strategic moves at Principal Engineer level

1. Build sanctioned paths faster than shadow IT can grow
2. Bring structured evaluation discipline to AI projects
3. Be the bridge to MRM and risk
4. Pick architectural battles, not tool battles
5. Develop opinions through real use

---

## 12. Vendor Application Layer

### The 7 layers (bottom to top)

1. **Silicon / accelerator** — NVIDIA GPUs, TPUs, custom ASICs. HBM 80–192GB per GPU. Supply constrained.
2. **Inference serving infrastructure** — vLLM, TensorRT-LLM, SGLang. Handles batching, KV cache, sharding, quantization.
3. **Foundation model provider** — Anthropic, OpenAI, Google, Meta, Mistral, Cohere, xAI, Chinese labs.
4. **Inference reseller / cloud abstraction** — AWS Bedrock, Azure OpenAI, GCP Vertex, Together AI, Groq, OpenRouter.
5. **Orchestration framework** — LangChain, LangGraph, LlamaIndex, Semantic Kernel, DSPy.
6. **Agent platform / harness** — Claude Code, Cursor, Copilot, Devin, Cline, Aider.
7. **Application / domain layer** — end-user product.

### Compute/memory asymmetry

Compute and memory cost concentrate at **layers 1–3**. Layers 4–7 are nearly free in absolute terms. Cost levers are at layer 3 and below — model choice, caching, batching.

### Where a Principal Engineer operates

- Advise at layer 3 (model approval)
- Procurement at layer 4 (Bedrock vs. Azure vs. direct)
- Standards at layer 5 (frameworks)
- Rollout at layer 6 (agent platforms)
- Deliverables at layer 7 (custom apps)

---

## 13. Wells Fargo Context

### Tachyon's actual position

Vertically integrated internal platform spanning layers 3–5 with layer 6 bolted on:

- **Layer 3** — hosts Llama, other open-weight models internally
- **Layer 4** — brokers Gemini (via GCP) and other commercial models
- **Layer 5** — RAG service, tools marketplace, shared agent components
- **Layer 6** — agent marketplace (prebuilt internal agents)
- **Governance wrapper** — request/justification process, MRM coordination, approved model list, key issuance

### Why this architecture

1. Data residency and contractual surface area collapse
2. MRM coverage at scale
3. Cost aggregation and visibility
4. Approval workflow as forcing function
5. Build-vs-buy optionality preserved

### The two parallel governance tracks

**Track A — Tech-to-Business (Tachyon's core domain):**
AI that touches customers, customer data, business decisions, or regulated processes. Full MRM validation, full risk review. High bar because blast radius is high.

**Track B — Tech-to-Tech (developer productivity):**
AI helping engineers write code, review PRs, query internal systems. Different justification process, calibrated to different risk profile. Downstream controls (code review, CI, deploy process) exist regardless of AI participation.

### Track B architecture

- **Layer 6:** GitHub Copilot (licensed devs), Devin, Microsoft Copilot (all employees via M365)
- **Layer 5 (bank investment):** MCP servers for internal services (Jira, SonarQube, Confluence), prompt libraries, custom plugins
- **Layers 3–4:** mostly invisible — vendor's choice

### Two populations

- **Developers (GitHub license):** full Copilot surface. Risk mitigated by code review, CI, SDLC.
- **All employees (Microsoft Copilot):** AI in Office/Teams. Risk: AI content treated as authoritative.

### Track interaction

- Track B outputs can feed Track A inputs (dev-written code deployed to customer system)
- Track A justifications often involve Track B work (building the Track A agent uses Copilot)
- **MCP layer is durable infrastructure** — serves both tracks, longest-leverage internal AI investment

---

## 14. Foundation Model Providers

### What a foundation provider does (4 integrated functions)

1. Model research and architecture
2. Training data acquisition and curation
3. Training execution (pretraining + post-training)
4. Inference serving

### What's inside a frontier model

- **Weights** — 100B–2T+ parameters. FP16/BF16 or FP8. A 500B-param model at FP16 is 1TB; at FP8, 500GB.
- **KV cache** — per request, scales with context length. Often larger than model weights in aggregate.
- **Attention computation** — core matrix multiplies. O(N²) naive, O(N) with FlashAttention.
- **Mixture of Experts (MoE)** — only subset of parameters activate per token. Memory stays high, compute drops.
- **Speculative decoding** — small draft model proposes, big model verifies in parallel. 2–3× throughput.

### Economics

- Training costs upfront and lumpy ($50M–$500M+ per frontier run)
- Inference margins thin and competitive
- Capacity supply-constrained (GPUs, power, cooling)
- Data moats eroding
- **No enduring moat at the model layer right now**

### What enterprises actually evaluate

1. Capability fit for use case (your task-specific evals, not vendor benchmarks)
2. Behavioral characteristics (hallucination, refusal patterns, agentic behavior)
3. Context window (nominal and effective)
4. Latency (TTFT, throughput)
5. Pricing structure (per-token, batch, cached, fine-tune)
6. Data handling and contractual terms
7. Versioning and deprecation policy
8. Regional availability
9. Compliance certifications
10. Operational maturity

### Where differentiation lives now

- Post-training methodology
- Tool use and agentic behavior
- Reasoning specialization
- Safety profile
- Multimodality

---

## 15. Provider Highlights

### Frontier closed-weight

- **Anthropic** — Claude (Opus, Sonnet, Haiku). Constitutional AI / RLAIF pioneer. Strong on coding, agentic tool use, instruction following. Conservative safety. Research-forward.
- **OpenAI** — GPT series, reasoning o-series. Broadest product surface. Microsoft partnership dominant channel. Pioneered test-time compute reasoning paradigm.
- **Google DeepMind** — Gemini series (Pro, Flash, Ultra, thinking variants). Vertically integrated with TPUs. Native multimodal. Strong long context.

### Open-weight providers

- **Meta (Llama)** — open weights at near-frontier capability. Strategic commoditization play. Default internal choice at most enterprises.
- **Mistral** — French. Open-weight + commercial. European data residency positioning.
- **DeepSeek** — Chinese. Caused disruption early 2025 with frontier-comparable capability at much lower training cost. Typically off bank approved lists.
- **Alibaba (Qwen)** — Chinese. Similar profile to DeepSeek.

### Specialists and next tier

- **Cohere** — enterprise-focused, strong on RAG and embeddings
- **xAI** — Grok, Musk's lab, bundled with X
- **AI21 Labs** — Jamba (hybrid Mamba+Transformer)
- **Aleph Alpha** — German, EU sovereignty positioning

### Pure-inference (layer 4)

- **Groq** — custom LPU silicon, very high throughput
- **Together AI, Fireworks, Anyscale** — multi-model inference platforms
- **OpenRouter** — meta-router for developers

### Strategic framings

1. Three frontier closed providers leapfrog each other — don't anoint a permanent winner
2. Open-weight follows closed-weight with 6–18 month lag
3. Specialization happens at application layer, not model layer
4. Geographic/political alignment matters (US default; Chinese typically off-list)
5. "Best model" question is wrong; "best for this workload at this price with these terms" is right

---

## 16. Theory: Tokenization

### The problem

Neural networks operate on numbers (vectors of real numbers). Text is not numbers.

- **Per-character** — sequences balloon, model learns word structure from scratch
- **Per-word** — unbounded vocabulary, out-of-vocabulary breaks, no subword structure
- **Subword (actual approach)** — chunks larger than characters, smaller than words. Fixed vocabulary (30K–200K) covers any input by composition.

### What a token is

An integer ID pointing to an entry in a vocabulary table. Vocabulary maps integer IDs to byte sequences (UTF-8 text fragments).

```
ID 15043 → " the"   (leading space is part of token)
ID 8884  → " walk"
ID 18108 → "ing"
ID 12379 → "Wells"
ID 33420 → " Fargo"
```

### Key properties

- Leading spaces are part of tokens
- Common words tend to be single tokens
- Rare words decompose into pieces
- Code and structured text tokenize differently (denser)
- Non-English languages often less efficient (more tokens per word)

### How tokenizers are created (BPE)

1. Start with all bytes as base tokens (256 for UTF-8)
2. Count most common adjacent pairs in corpus
3. Merge most common pair into a new token
4. Repeat until target vocabulary size reached

Pure frequency statistics. No semantic meaning in the choices.

### Every vendor tokenizes differently

- Different training corpora
- Different vocabulary sizes
- Different algorithm choices (BPE, WordPiece, SentencePiece)
- Different language priorities

**Practical consequences:**

1. Same text consumes different token counts across vendors
2. Tokenization affects behavior subtly (character counting failures)
3. Tokens cannot be transferred between models
4. Token counts in tools are tokenizer-specific
5. Prompt caching is tokenizer-locked

### How tokens relate — the answer

**At the tokenizer level:** not at all. Integer IDs are arbitrary labels.

**At the embedding layer and above:** through learned high-dimensional vectors. Tokens with similar function in language end up with similar embedding vectors after training — because making them similar improved next-token prediction during training.

Tokens are the keys; embeddings are the values; meaning emerges at the embedding layer.

### Flow through the network

**Input:**
1. User text → tokenizer → integer IDs
2. Each ID looks up embedding vector in embedding table
3. Vectors flow through transformer layers
4. Final vector at each position

**Output:**
5. Final vector at last position → projection through unembedding matrix
6. Produces logits: one score per vocabulary token
7. Softmax → probability distribution
8. Sampling picks one token
9. Picked token appended to sequence
10. Repeat for next token

The model generates one token at a time. A 500-word response = 600–700 forward passes.

### Wells Fargo framings

1. Token counting is a discipline — do it in target model's actual tokens
2. Comparing providers requires tokenizer-aware comparison
3. Surprising LLM behaviors often trace to tokenization
4. Integer-IDs-don't-mean-anything is the bridge to embeddings

---

## 17. Theory: Attention Mechanics

### The problem attention solves

The model producing the next token needs information from earlier tokens — sometimes many tokens back. Attention is the mechanism that does this lookup.

### Q, K, V — the three vectors

For every token, the model computes three vectors derived from that token's current representation:

- **Q (query)** — "what am I looking for in the past?" (the asker)
- **K (key)** — "how do I describe myself to anyone looking back at me?"
- **V (value)** — "if someone decides I'm relevant, here's what I contribute"

### Database analogy

Query matched against keys (indexes) returns values (data). Attention works the same, but with continuous similarity (dot products) and weighted sums over all candidates instead of exact matches.

### The mechanic

Generating token #1000:

1. Compute token #1000's Q (what it's looking for)
2. Compare Q against every prior token's K (tokens #1–999) via dot product → 999 similarity scores
3. Softmax the scores → attention weights (sum to 1)
4. Weighted sum of prior tokens' V vectors using those weights
5. Result is the attention output for token #1000

Many attention heads in parallel per layer. Many layers stacked. Each head learns to look for different patterns.

### What the model "learns"

The W_Q, W_K, W_V matrices start random. Through training, they're shaped to produce Q/K/V vectors that result in useful attention patterns for predicting the next token. Different heads specialize in syntax, entities, formatting, etc. — all emergent.

### Why Q is not cached but K and V are

- **Q is used exactly once.** Token #1000's Q matters only when generating token #1000. Caching it is pure waste.
- **K and V get used over and over.** Token #500's K and V are referenced by every subsequent token being generated — #501, #502, …, forever.

**The asymmetry:** Q is the asker (used once). K and V are the answerer (used forever). Cache the long-lived role, not the single-use role.

### Causal masking

Attention is one-directional — token #1000 can only look at tokens #1–999. This makes caching possible: K and V for prior tokens are permanently correct because nothing after can change them.

### Why attention is expensive

For context length N: N × N similarity scores per layer, per head. Naive scaling O(N²). Doubling context more than doubles compute.

**FlashAttention** rewrites the computation to be block-wise, reducing memory traffic dramatically. Same FLOPs, much better real-world performance.

### How attention explains failure modes

| Failure | Attention explanation |
|---------|----------------------|
| Lost-in-the-middle | Position biases + softmax competitive dilution over many Ks |
| Context rot | Noise Ks compete with signal Ks for attention weight |
| Summary drift | Compaction deletes K vectors for info not in summary |
| Tool thrashing | Unfocused Q produces flat attention → muddy output → more context |
| Premature convergence | Model's own prior outputs create strong Ks reinforcing framing |
| Prompt injection | No architectural distinction between trusted/untrusted Ks |
| Poisoned context | Original wrong Ks persist, compete with correction Ks |

### Unified framing

*"Attention is content-addressable memory with no garbage collection. Everything you put in stays in, competing for weight, until you start fresh. Context discipline isn't optional — it's the only mechanism we have to keep the attention signal clean."*

---

## 18. Theory: RLHF and RLAIF

Training happens in stages. Different stages produce qualitatively different things.

### Stage 1: Pretraining
Shown trillions of tokens. Objective: predict next token. Produces **base model** — capable but not useful as assistant. If asked "What is the capital of France?", base model might complete with "What is the capital of Germany? What is the capital of Italy?" — plausible text continuation, not an answer.

### Stage 2: Supervised Fine-Tuning (SFT)
Trained on curated example conversations. Model learns the shape of being a helpful assistant. Necessary but not sufficient.

### Stage 3: RLHF — Reinforcement Learning from Human Feedback

1. Model generates two responses to same prompt
2. Human labelers pick which is better
3. A separate **reward model** is trained on human preference data
4. Main LLM is trained via RL (PPO or similar) against the reward model

The reward model is a learned proxy for human judgment. Main model learns to produce outputs the reward model scores highly.

**Core trick:** lets you optimize for things hard to express as a loss function. "Be helpful" isn't a mathematical objective, but "produce outputs humans rate as more helpful" is.

**Challenges:**
- Expensive (hundreds of thousands to millions of comparisons)
- Labelers disagree
- **Reward hacking** — model learns outputs that score high but are actually subtly worse
- Labeling guideline quality determines model values

### Stage 4: RLAIF — Reinforcement Learning from AI Feedback

Same structure, but AI model does the comparisons using written principles (a "constitution"). Core insight behind **Anthropic's Constitutional AI**.

**Advantages:**
- Orders of magnitude cheaper
- More consistent than human labeling
- Auditable (principles are written down)
- Scalable to dimensions humans struggle with

**Concerns:**
- AI labeler has its own biases
- Written principles still encode choices
- Potential drift from human values

Modern frontier models use **combinations** of SFT + RLHF + RLAIF. The mix differentiates providers.

### Wells Fargo framings

1. Post-training is where personality lives — pretrained capability converges; post-training differentiates
2. Refusal patterns come from post-training — learned behavior, not capability limit
3. Bank-specific fine-tuning is a post-training operation — SFT or RLHF on top of provider's base
4. Reward hacking is a real failure mode — look past surface plausibility

---

## 19. Theory: FLOPs

- **FLOP** = Floating-Point Operation (one arithmetic op)
- **FLOPs** = plural, count of total work done
- **FLOPS** = FLOPs per second, a rate

An NVIDIA H100 does ~10^15 FLOPS at FP16 (one petaFLOPS).

### Why count FLOPs in AI

1. Estimate compute cost (training = 10^25–10^26 FLOPs for frontier)
2. Compare model sizes meaningfully (FLOPs/token > parameter count)
3. Predict hardware requirements
4. Track scaling laws (capability scales predictably with training FLOPs)

### Rule of thumb for dense transformers

Forward pass FLOPs ≈ 2 × parameters × tokens

A 100B-parameter model generating one token: ~2 × 10^11 FLOPs. On an H100: 0.2ms of pure arithmetic. Real latency is higher because memory access dominates, not arithmetic.

---

## 20. Theory: Sampling and Temperature

### Logits
After the final layer: a vector of logits (one number per vocabulary token). Unbounded real values. Higher = more likely according to model.

### Softmax
Converts logits to probability distribution:
1. Exponentiate every logit (forces positive, amplifies differences)
2. Normalize (divide by sum → sums to 1)

Produces model's actual prediction — "given this context, probability distribution over what comes next."

### Sampling strategies

**Greedy decoding (temp = 0):** Always pick highest. Deterministic. Can cause repetition loops. Misses good alternatives.

**Pure random sampling:** Pick with probabilities matching distribution. Occasionally picks bizarre rare tokens.

**Top-k sampling:** Keep only top k tokens, renormalize, sample.

**Top-p (nucleus) sampling:** Keep tokens until cumulative probability reaches p (e.g., 0.9). Adaptive — sharp distributions have small nucleus, flat have large. **Most common in modern frontier models.**

### Temperature

Modifies logits before softmax:
- **T = 1.0** — default, raw predictions
- **T < 1.0** — sharpens distribution (more focused, more predictable)
- **T > 1.0** — flattens distribution (more diverse, less coherent)
- **T = 0** — degenerate case = greedy

### The key insight

**The "intelligence" of an LLM's response is the shape of the probability distribution, not the specific sampled token.**

A good model produces sharp accurate distributions on clear questions, spreads probability across valid options on open questions, flattens when it doesn't know. Sampling just rolls dice over whatever distribution the model produces.

### Wells Fargo framings

1. Temperature is a deployment decision, not just a toggle
2. Non-determinism affects testing — run T=0 for reproducibility, or many samples for distribution
3. "The model said X" understates — more precise: "the model produced a distribution that sampled to X"
4. Some sampling failures look like model failures — low-probability early token biases everything after

### Recommended temperature by task

| Task | Temperature |
|------|------------|
| Structured extraction (credit reports, loan fields) | 0 or 0.1 |
| Classification | 0 or 0.1 |
| Code generation (specified) | 0.1–0.3 |
| Code generation (exploratory) | 0.5–0.8 |
| Document summarization | 0.3–0.7 |
| Email/response drafting | 0.7–1.0 |
| Brainstorming | 1.0+ |
| Customer-facing copy | 0.3–0.5 |

### The bank-specific framing

*"Temperature should be chosen for the use case, not as a default. For regulated outputs where the same input should produce the same output for auditability, set temperature to 0 and document it. Treat temperature as a deployment parameter that goes in your model card."*

---

## Appendix: Key Mental Models to Carry

1. **LLM stateless, harness stateful** — statefulness is a design choice made in client code, not a model property
2. **Context is working memory, not permanent storage** — fills up as the agent acts
3. **Attention is content-addressable memory with no garbage collection** — context discipline is the only mechanism to keep signal clean
4. **The LLM provides judgment; the harness provides execution and policy** — know which is doing what
5. **Q is asker (used once); K and V are answerers (used forever)** — asymmetric caching
6. **Tokens are keys; embeddings are values; meaning lives in the embedding space** — tokens themselves are arbitrary integers
7. **The distribution is the model's output; the sampled token is bookkeeping** — intelligence lives in the shape
8. **Post-training is where personality lives** — pretrained capability converges; post-training differentiates
9. **In regulatory language, an LLM is a model; MRM applies** — first question on any AI pitch: what tier?
10. **The harness, not the model, is where vendor lock-in lives** — strategic decisions should target the harness layer

---

*End of study guide.*
