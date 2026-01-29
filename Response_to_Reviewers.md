# Response to Reviewers

We would like to thank the reviewers for their insightful feedback and constructive criticism. We have carefully revised the manuscript to address all concerns. Below, we detail the specific changes made in response to each reviewer's comments.

## Response to Reviewer 1

**Comment 1.1:** "The discussion of how supervision signals are obtained for table-text alignment is relatively brief."
**Response:** We have expanded **Section 3.2.2 (Edge Formation)** to explicitly detail the supervision signals. Specifically, we now define the exact cosine similarity threshold ($\tau=0.75$) used to establish semantic edges between paragraphs and table rows. We also expanded the description of the supervised contrastive loss in **Section 3.3**, explaining how positive pairs (aligned text-cell) are contrasted against negative samples to optimize the graph representation.

**Comment 1.2:** "For the routing logic, the actual decision mechanism remains somewhat abstract."
**Response:** We have formalized the routing logic by adding **Algorithm 1 (Symbolic-Neural Fusion Routing)** in Section 3.5. This algorithm step-by-step details how the system classifies queries into *Neural*, *Symbolic*, or *Hybrid* modes based on probabilistic scores and context types. We also clarified the terminology, now referring to it as "**Probabilistic Hard-Routing**" to distinguish it from soft-gating approaches.

**Comment 1.3:** "A substantial portion of the bibliography consists of arXiv preprints... replace with peer-reviewed versions."
**Response:** We have conducted a comprehensive audit of our bibliography. We have replaced arXiv citations with their peer-reviewed counterparts where available (e.g., *FinQA* is now cited as EMNLP 2021, *RAG* as NeurIPS 2020). We also prioritized foundational references for graph neural networks (citing Veličković et al., ICLR 2018 for GAT).

---

## Response to Reviewer 2

**Comment 2.1:** "The paper would benefit from a more detailed and transparent description of the methodology."
**Response:** We have significantly elaborated on the methodology in Section 3.
*   **Graph Construction:** We added **Table 1 (Summary of Notations)** and a detailed breakdown of node/edge types.
*   **Architecture Visualization:** We added **Figure 1**, a comprehensive system diagram illustrating the three-phase pipeline (Indexing, Retrieval, Fusion).

**Comment 2.2:** "The experimental section is promising but... deeper analysis, both qualitative and quantitative, would strengthen the validity."
**Response:** We have bolstered the experimental section significantly:
*   **Qualitative Analysis:** We added a new **Qualitative Case Study (Table 3)**, which provides a step-by-step walkthrough of a complex query ("R&D expenses excluding stock-based compensation"). This table contrasts our graph traversal approach against a specific Vanilla RAG failure mode.
*   **Routing Analysis:** We added **Figure 5** and the accompanying "Analysis of Probabilistic Routing" subsection. This quantitatively breaks down performance by mode, demonstrating that our *Hybrid* mode achieves 88.4% accuracy on the most complex query subset.

**Comment 2.3:** "More clearly articulate its original contribution."
**Response:** We have completely rewritten the **Introduction** to explicitly list our contributions: (1) Structure-Aware Graph Construction, (2) The TTGNN architecture, and (3) The Probabilistic Hard-Routing mechanism. We moved the "Related Work" comparison to highlight these differentiators early on.

---

## Response to Reviewer 3

**Comment 3.1:** "The paper is hard to follow... The English needs a lot of improvement."
**Response:** We have rewritten the **Introduction (Section 1)** to use simpler, more direct language. We focused on clearly defining the problem (financial "haystacks") and our solution without unnecessary jargon.

**Comment 3.2:** "Add a proposed model diagram."
**Response:** We have added **Figure 1**, a detailed architectural diagram visualizing the flow from PDF parsing to Graph Construction to Hybrid Reasoning.

**Comment 3.3:** "Add a table to compare between related works."
**Response:** We have included a comparison table in the Related Work section (referenced in the revised manuscript) that contrasts HierFinRAG with Vanilla RAG, Generic GraphRAG, and Agentic RAG across key dimensions like "Structure Awareness" and "Inference Latency".

**Comment 3.4:** "Add algorithm template."
**Response:** We added **Algorithm 1** in Section 3.5.

**Comment 3.5:** "FinQA and FinanceBench were referenced too late."
**Response:** We have moved the introduction of these datasets to the **Introduction** section, establishing the evaluation context effectively on page 1.

**Comment 3.6:** "Structure as a First-Class Citizen - what do you mean?"
**Response:** We have clarified this concept in the **Discussion (Section 5.2)**. It refers to our design choice of explicitly modeling document layout (headers, sections) as graph nodes, rather than treating the document as a flat sequence of text. We cite our Ablation Study and Case Study as empirical evidence of its importance.
