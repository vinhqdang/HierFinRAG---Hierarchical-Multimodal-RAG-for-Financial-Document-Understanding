# Technical Research Proposal: HierFinRAG - Hierarchical Multimodal RAG for Financial Document Understanding

## 1. Research Problem and Motivation

### 1.1 The Performance Gap

Current state-of-the-art systems demonstrate significant limitations on financial document understanding:

- **FinQA**: GPT-4 achieves 76% EM accuracy vs 89-91% human performance (13-15 point gap)
- **FinanceBench**: RAGAS framework fails on 83.5% of questions involving numerical data
- **Root Cause**: Existing RAG systems either:
  1. Flatten tables to text → Loses structure, poor numerical reasoning
  2. Treat tables as separate modality → Misses table-text integration and cross-references
  3. Use pure vision-based approaches → Requires massive training data, computationally expensive

### 1.2 Core Challenge

Financial documents (10-K/10-Q filings, earnings reports) require **simultaneous understanding** across multiple modalities:

1. **Narrative sections**: MD&A, risk factors, executive commentary
2. **Structured tables**: Balance sheet, income statement, cash flow statement
3. **Footnote cross-references**: Links between narrative text and specific table entries
4. **Charts and graphs**: Visual trends complementing tabular data
5. **Temporal relationships**: Multi-period comparisons, YoY/QoQ analysis
6. **Accounting constraints**: Assets = Liabilities + Equity, conservation principles

**Research Hypothesis**: A unified hierarchical architecture that jointly models table-text relationships through graph neural networks, combined with symbolic reasoning for numerical operations, can close the 60-80% of the human-AI performance gap on complex financial reasoning tasks.

---

## 2. Proposed Algorithm: HierFinRAG

### 2.1 System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    FINANCIAL DOCUMENT INPUT                      │
│              (10-K, 10-Q, Earnings Report, etc.)                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│               STAGE 1: STRUCTURE-AWARE PARSING                   │
│  ┌──────────────┬──────────────┬──────────────┬───────────────┐ │
│  │   Narrative  │   Tables     │   Charts     │ Cross-Refs    │ │
│  │   Extraction │   Extraction │   Extraction │   Extraction  │ │
│  └──────┬───────┴──────┬───────┴──────┬───────┴──────┬────────┘ │
└─────────┼──────────────┼──────────────┼──────────────┼──────────┘
          │              │              │              │
          ▼              ▼              ▼              ▼
┌─────────────────────────────────────────────────────────────────┐
│         STAGE 2: MULTIMODAL KNOWLEDGE GRAPH CONSTRUCTION         │
│                                                                   │
│   Nodes: [Paragraph] [Section] [Table] [Cell] [Chart Element]  │
│                                                                   │
│   Edges:                                                         │
│   • Semantic Similarity (cosine between embeddings)             │
│   • Structural Hierarchy (section contains paragraph)           │
│   • Cross-References ("See Note 5", "Table 2.1")               │
│   • Temporal Links (2023 vs 2024 comparisons)                   │
│   • Accounting Relations (cell dependencies)                     │
│                                                                   │
│              ┌─────────────────────────────┐                    │
│              │  TABLE-TEXT GRAPH NEURAL    │                    │
│              │  NETWORK (TTGNN) ENCODER    │                    │
│              └─────────────────────────────┘                    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 3: QUERY PROCESSING PIPELINE                  │
│                                                                   │
│  User Query → Financial Intent Classifier                        │
│               ↓                                                   │
│    ┌──────────┴─────────────────────────────────────┐          │
│    │                                                   │          │
│  Intent_Narrative → Text Retrieval                    │          │
│  Intent_Numerical → Table Retrieval + Calculator      │          │
│  Intent_Trend     → Chart Retrieval + Symbolic        │          │
│  Intent_Compare   → Multi-Period Aggregation          │          │
│                                                                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│         STAGE 4: HIERARCHICAL ATTENTION RETRIEVAL                │
│                                                                   │
│  Level 1: DOCUMENT SELECTION                                    │
│  ├─ Coarse-grained: Which sections (MD&A, Financials, Notes)?  │
│  │  Top-k sections based on query-section similarity            │
│  │                                                               │
│  Level 2: SECTION DECOMPOSITION                                 │
│  ├─ Medium-grained: Which paragraphs/tables within section?    │
│  │  Joint retrieval across text and tabular elements            │
│  │                                                               │
│  Level 3: FINE-GRAINED EXTRACTION                               │
│  └─ Which specific sentences/cells contain the answer?          │
│     Cell-level retrieval for numerical queries                   │
│                                                                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│          STAGE 5: SYMBOLIC-NEURAL FUSION REASONING               │
│                                                                   │
│  ┌──────────────────────┐    ┌─────────────────────────┐       │
│  │   NEURAL COMPONENT    │    │  SYMBOLIC COMPONENT      │       │
│  │  • Ambiguity handling │    │  • Exact arithmetic      │       │
│  │  • Implicit inference │    │  • Constraint checking   │       │
│  │  • Cross-doc synthesis│    │  • Formula evaluation    │       │
│  └──────────┬────────────┘    └──────────┬──────────────┘       │
│             │                             │                       │
│             └──────────┬──────────────────┘                       │
│                        ▼                                          │
│              Fused Reasoning Module                               │
│              • Decides when to use symbolic vs neural            │
│              • Verifies numerical answers against constraints    │
│              • Resolves conflicts between modalities             │
│                                                                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│        STAGE 6: ANSWER GENERATION WITH ATTRIBUTION               │
│                                                                   │
│  Generated Answer                                                │
│  + Supporting Table Cells (coordinates + values)                 │
│  + Supporting Text Sentences                                     │
│  + Reasoning Chain (step-by-step)                               │
│  + Confidence Score                                              │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Stage 1: Structure-Aware Parsing

**Objective**: Extract and preserve structural information from financial PDFs

**Components**:

1. **Document Layout Analysis**
   - **Tool**: LayoutLMv3 or Docling (IBM's document parsing)
   - **Output**: Bounding boxes for text blocks, tables, figures with hierarchical structure
   - **Preservation**: Maintain parent-child relationships (Section → Subsection → Paragraph)

2. **Table Extraction and Understanding**
   - **Tool**: Table Transformer or PaddleOCR's table recognition
   - **Process**:
     - Detect table boundaries
     - Extract cell contents and coordinates (row, column)
     - Identify header rows and columns
     - Parse merged cells and multi-line headers
   - **Output**: Structured representation: `Table(id, caption, headers, cells[row][col])`
   
3. **Cross-Reference Detection**
   - **Pattern Matching**: Regular expressions for common patterns
     - "See Note X", "Table X.Y", "as described in", "referenced above"
   - **Entity Linking**: Map textual mentions to specific table IDs or section IDs
   - **Output**: Graph edges connecting referencing text to referenced element

4. **Chart and Figure Processing**
   - **Tool**: DETR or YOLOv8 for figure detection
   - **Caption Extraction**: OCR on figure captions
   - **Placeholder Strategy**: Store figure metadata (type, caption, position)
   - **Future Enhancement**: Visual question answering over charts

**Technical Specifications**:
- **Input Format**: PDF documents
- **Intermediate Format**: JSON structure maintaining hierarchy
- **Performance Target**: Parse 100-page 10-K in <60 seconds

### 2.3 Stage 2: Table-Text Graph Neural Network (TTGNN)

**Core Innovation**: Unified graph representation enabling joint reasoning across modalities

**Node Types and Embeddings**:

1. **Paragraph Nodes (P)**
   - **Embedding**: `E5-Mistral-7B` or `OpenAI text-embedding-3-large`
   - **Dimension**: 1024-dim or 3072-dim
   - **Additional Features**: 
     - Section type (MD&A, Risk Factors, Financials)
     - Position in document (normalized)
     - Length (token count)

2. **Section Header Nodes (S)**
   - **Embedding**: Same encoder as paragraphs
   - **Role**: Hierarchical anchors for coarse-grained retrieval
   - **Additional Features**: 
     - Section level (1, 2, 3, etc.)
     - Section ID (for cross-references)

3. **Table Nodes (T)**
   - **Embedding**: Table caption + concatenated column headers
   - **Dimension**: Same as text nodes for unified space
   - **Additional Features**:
     - Number of rows, columns
     - Table type classification (Balance Sheet, Income Statement, Custom)
     - Time period covered

4. **Cell Nodes (C)**
   - **Embedding**: Row header + Column header + Cell value
   - **Special Handling**: 
     - Numerical cells: Include both text value and parsed float
     - Text cells: Standard text embedding
   - **Additional Features**:
     - Cell position (row_idx, col_idx)
     - Data type (currency, percentage, date, text)
     - Cell dependencies (for formulas)

5. **Chart Element Nodes (G)** (Optional for future work)
   - **Embedding**: Chart caption + type
   - **Placeholder**: For vision-language integration

**Edge Types and Construction**:

1. **Semantic Similarity Edges (E_sem)**
   - **Computation**: Cosine similarity between node embeddings
   - **Threshold**: Connect if similarity > 0.7
   - **Weight**: w_sem = cosine_similarity(emb_i, emb_j)
   - **Pruning**: Keep only top-k neighbors per node to limit graph density

2. **Structural Hierarchy Edges (E_struct)**
   - **Direction**: Parent → Child
   - **Examples**:
     - Section → Paragraph
     - Table → Cell
     - Document → Section
   - **Weight**: Fixed w_struct = 1.0 (always strong connection)

3. **Cross-Reference Edges (E_ref)**
   - **Detection**: Pattern matching + entity resolution
   - **Direction**: Referring text → Referenced element
   - **Weight**: w_ref = 1.0 (explicit mention)
   - **Examples**:
     - Paragraph mentions "See Note 5" → Link to Note 5 section
     - Text mentions "Table 2.1" → Link to Table 2.1

4. **Temporal Edges (E_temp)**
   - **Connect**: Same metric across different time periods
   - **Examples**:
     - Revenue_2023 → Revenue_2024
     - Q1_Sales → Q2_Sales
   - **Weight**: w_temp = 0.8 (implicit temporal relationship)

5. **Accounting Relation Edges (E_acc)**
   - **Based On**: Known accounting identities
   - **Examples**:
     - Total_Assets → [Current_Assets, Non_Current_Assets] (summation)
     - Net_Income → [Revenue, Expenses] (subtraction)
   - **Weight**: w_acc = 1.0 (hard constraint)
   - **Purpose**: Enable constraint verification in symbolic reasoning

**Graph Neural Network Architecture**:

```python
class TTGNN(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, num_layers=3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Node type embeddings
        self.node_type_emb = nn.Embedding(5, hidden_dim)  # P, S, T, C, G
        
        # Edge type embeddings
        self.edge_type_emb = nn.Embedding(5, hidden_dim)  # sem, struct, ref, temp, acc
        
        # Graph attention layers
        self.gat_layers = nn.ModuleList([
            GATv2Conv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                heads=8,
                edge_dim=hidden_dim,
                concat=False
            ) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x, edge_index, edge_attr, node_types):
        """
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge types [num_edges]
            node_types: Node type IDs [num_nodes]
        
        Returns:
            Updated node embeddings [num_nodes, hidden_dim]
        """
        # Add node type information
        h = x + self.node_type_emb(node_types)
        
        # Message passing
        for gat in self.gat_layers:
            edge_embeddings = self.edge_type_emb(edge_attr)
            h = gat(h, edge_index, edge_attr=edge_embeddings)
            h = F.relu(h)
            h = F.dropout(h, p=0.1, training=self.training)
        
        # Output projection
        h = self.output_proj(h)
        
        return h
```

**Training Strategy**:

1. **Pre-training Task**: Masked Node Prediction
   - Mask 15% of nodes
   - Predict original embeddings from context
   - Loss: MSE between predicted and original embeddings

2. **Contrastive Learning**: Node-Pair Similarity
   - Positive pairs: Nodes connected by cross-reference edges
   - Negative pairs: Random nodes from different documents
   - Loss: InfoNCE contrastive loss

3. **Supervised Fine-tuning**: Question-Answer Pairs
   - Given query embedding, retrieve top-k relevant nodes
   - Maximize similarity with gold supporting nodes
   - Loss: Cross-entropy over node relevance scores

**Graph Construction Details**:
- **Average Graph Size**: 10K-50K nodes per 10-K document
- **Edge Density**: Sparse (avg degree ~10-20)
- **Memory Optimization**: Mini-batch training with graph sampling (GraphSAINT or NeighborSampler)

### 2.4 Stage 3: Financial Intent Classification

**Purpose**: Determine query type to route to appropriate retrieval strategy

**Intent Categories**:

1. **Narrative** (40% of queries): "What are the company's primary risk factors?"
   → Text-focused retrieval from MD&A, Risk Factors sections

2. **Numerical** (30% of queries): "What was the total revenue in 2023?"
   → Table-focused retrieval + calculator tool invocation

3. **Trend** (15% of queries): "How has operating margin changed over the past 3 years?"
   → Multi-period table retrieval + temporal reasoning

4. **Comparison** (10% of queries): "Compare R&D expenses between 2023 and 2024"
   → Multi-cell retrieval + arithmetic operations

5. **Hybrid** (5% of queries): "Explain why revenue decreased despite growth in units sold"
   → Combined text-table retrieval + causal reasoning

**Model Architecture**:
- **Base Model**: DeBERTa-v3-large fine-tuned on financial queries
- **Input**: Query text
- **Output**: Intent probability distribution [5 classes]
- **Multi-Label**: Queries can have multiple intents (primary + secondary)

**Training Data Construction**:
- **Manual Labeling**: 2,000 queries from FinQA, ConvFinQA, FinanceBench
- **Augmentation**: GPT-4 generates 5,000 additional queries with intent labels
- **Validation**: Human expert review of 500 samples for quality

**Performance Requirement**: >90% accuracy on intent classification

### 2.5 Stage 4: Hierarchical Attention Retrieval

**Key Insight**: Multi-level retrieval is more efficient and accurate than flat retrieval

**Level 1: Document-Level Selection (Coarse-Grained)**

**Objective**: Identify top-k relevant sections from entire document

**Method**:
```python
def level1_retrieval(query_emb, section_embeddings, k=5):
    """
    Args:
        query_emb: Query embedding [1, d]
        section_embeddings: All section embeddings [num_sections, d]
        k: Number of sections to retrieve
    
    Returns:
        top_k_sections: Indices of most relevant sections
    """
    # Compute cosine similarity
    similarities = cosine_similarity(query_emb, section_embeddings)
    
    # Get top-k sections
    top_k_indices = torch.topk(similarities, k=k).indices
    
    return top_k_indices
```

**Section Candidates**:
- Business Overview
- Risk Factors
- Management Discussion & Analysis (MD&A)
- Financial Statements (Balance Sheet, Income, Cash Flow)
- Notes to Financial Statements
- Management Compensation
- Related Party Transactions

**Level 2: Section-Level Decomposition (Medium-Grained)**

**Objective**: Within selected sections, identify relevant paragraphs and tables

**Method**: Hybrid retrieval combining sparse and dense

```python
def level2_retrieval(query, section_nodes, k=10):
    """
    Args:
        query: Query text
        section_nodes: List of [Paragraph, Table] nodes in selected sections
        k: Number of elements to retrieve
    
    Returns:
        top_k_elements: Most relevant paragraphs/tables
    """
    # Sparse retrieval (BM25)
    bm25_scores = bm25_retriever.score(query, section_nodes)
    
    # Dense retrieval (embedding similarity)
    query_emb = embed_model.encode(query)
    dense_scores = cosine_similarity(query_emb, [n.embedding for n in section_nodes])
    
    # Reciprocal Rank Fusion
    combined_scores = reciprocal_rank_fusion(bm25_scores, dense_scores)
    
    # Get top-k
    top_k_indices = np.argsort(combined_scores)[-k:][::-1]
    
    return [section_nodes[i] for i in top_k_indices]
```

**Reciprocal Rank Fusion (RRF)**:
```
RRF(d) = Σ_r 1/(k + rank_r(d))
where r iterates over different ranking sources, k=60 (standard)
```

**Level 3: Cell-Level Extraction (Fine-Grained)**

**Objective**: For numerical queries, identify specific table cells

**Method**: Two-stage approach

**Stage 3a: Row Selection**
```python
def select_relevant_rows(query, table, top_k=5):
    """
    Args:
        query: Query text
        table: Table object with rows
        top_k: Number of rows to select
    
    Returns:
        relevant_rows: Top-k rows by relevance
    """
    row_representations = []
    for row in table.rows:
        # Concatenate row header + all cell values
        row_text = row.header + " " + " ".join([c.value for c in row.cells])
        row_representations.append(row_text)
    
    # Embed and compute similarity
    query_emb = embed_model.encode(query)
    row_embs = embed_model.encode(row_representations)
    similarities = cosine_similarity(query_emb, row_embs)
    
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]
    return [table.rows[i] for i in top_k_indices]
```

**Stage 3b: Column Selection**
```python
def select_relevant_columns(query, table, intent):
    """
    Args:
        query: Query text
        table: Table object
        intent: Financial intent (helps identify column type)
    
    Returns:
        relevant_columns: Columns likely to contain answer
    """
    if intent == "Numerical":
        # For specific year mentions, select that year's column
        year_match = re.search(r'(20\d{2})', query)
        if year_match:
            year = year_match.group(1)
            matching_cols = [col for col in table.columns if year in col.header]
            if matching_cols:
                return matching_cols
        
        # Otherwise, select most recent period columns
        return table.get_latest_period_columns()
    
    elif intent == "Comparison":
        # Select multiple period columns
        return table.get_time_series_columns()
    
    else:
        # Default: all columns
        return table.columns
```

**Final Cell Extraction**:
```python
def extract_answer_cells(relevant_rows, relevant_columns):
    """
    Returns:
        cells: Intersection of relevant rows and columns
    """
    cells = []
    for row in relevant_rows:
        for col in relevant_columns:
            cell = table.get_cell(row.index, col.index)
            cells.append(cell)
    
    return cells
```

**Hierarchical Attention Benefits**:
1. **Efficiency**: Search space reduces from millions → thousands → dozens
2. **Accuracy**: Coarse filtering removes irrelevant content early
3. **Interpretability**: Can trace decision at each level
4. **Flexibility**: Can stop at any level based on query complexity

### 2.6 Stage 5: Symbolic-Neural Fusion Reasoning

**Core Challenge**: Financial reasoning requires both neural flexibility (understanding ambiguity) and symbolic precision (exact arithmetic)

**Architecture**:

```python
class SymbolicNeuralFusion(nn.Module):
    def __init__(self, llm_model="gpt-4o", calculator_precision=2):
        self.llm = llm_model  # Neural component
        self.calculator = SymbolicCalculator(precision=calculator_precision)
        self.constraint_checker = AccountingConstraintChecker()
        self.router = ReasoningRouter()  # Decides which component to use
        
    def forward(self, query, retrieved_context):
        """
        Args:
            query: User question
            retrieved_context: Retrieved paragraphs, tables, cells
        
        Returns:
            answer: Final answer with reasoning trace
        """
        # 1. Route to appropriate reasoning mode
        reasoning_mode = self.router.determine_mode(query, retrieved_context)
        
        if reasoning_mode == "symbolic_only":
            # Pure calculation (e.g., "What is 2023 revenue + 2023 cost of goods sold?")
            answer = self.calculator.compute(query, retrieved_context)
            
        elif reasoning_mode == "neural_only":
            # Pure text understanding (e.g., "Summarize risk factors")
            answer = self.llm.generate(query, retrieved_context)
            
        elif reasoning_mode == "hybrid":
            # Most common case: Extract values neurally, compute symbolically
            
            # Step 1: LLM extracts structured information
            extraction_prompt = f"""
            Given the query: {query}
            And the retrieved context: {retrieved_context}
            
            Extract the following information:
            1. What values are needed to answer the query?
            2. What operation should be performed (add, subtract, divide, compare)?
            3. Are there any temporal qualifiers (YoY, QoQ)?
            
            Output in JSON format.
            """
            extraction = self.llm.generate(extraction_prompt)
            
            # Step 2: Map extracted values to actual cells
            value_mapping = self.map_values_to_cells(extraction, retrieved_context)
            
            # Step 3: Symbolic calculator performs operation
            numeric_result = self.calculator.compute(
                operation=extraction["operation"],
                values=value_mapping
            )
            
            # Step 4: Constraint checking
            is_valid = self.constraint_checker.verify(numeric_result, retrieved_context)
            
            if not is_valid:
                # Backtrack and try alternative interpretation
                answer = self.handle_constraint_violation(
                    query, retrieved_context, numeric_result
                )
            else:
                # Step 5: LLM generates natural language answer
                answer_prompt = f"""
                Query: {query}
                Computed result: {numeric_result}
                
                Generate a natural language answer that:
                1. States the result clearly
                2. Cites the specific table cells used
                3. Shows the calculation steps
                """
                answer = self.llm.generate(answer_prompt)
        
        return answer
```

**Symbolic Calculator Component**:

```python
class SymbolicCalculator:
    def __init__(self, precision=2):
        self.precision = precision
        
    def compute(self, operation, values):
        """
        Args:
            operation: String ("add", "subtract", "divide", "percentage_change", etc.)
            values: Dict mapping variable names to numeric values
        
        Returns:
            Computed result with proper precision
        """
        if operation == "add":
            result = sum(values.values())
            
        elif operation == "subtract":
            if len(values) == 2:
                result = list(values.values())[0] - list(values.values())[1]
            else:
                raise ValueError("Subtraction requires exactly 2 values")
                
        elif operation == "divide":
            numerator = values.get("numerator")
            denominator = values.get("denominator")
            if denominator == 0:
                raise ValueError("Division by zero")
            result = numerator / denominator
            
        elif operation == "percentage_change":
            old_value = values.get("old_value") or values.get("previous")
            new_value = values.get("new_value") or values.get("current")
            result = ((new_value - old_value) / old_value) * 100
            
        elif operation == "ratio":
            numerator = values.get("numerator")
            denominator = values.get("denominator")
            result = numerator / denominator
            
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        # Apply precision
        return round(result, self.precision)
```

**Accounting Constraint Checker**:

```python
class AccountingConstraintChecker:
    def __init__(self):
        self.constraints = [
            ("Assets", "=", "Liabilities + Equity"),  # Balance sheet identity
            ("Net_Income", "=", "Revenue - Expenses"),  # Income statement
            ("Cash_End", "=", "Cash_Begin + Cash_Inflow - Cash_Outflow"),  # Cash flow
        ]
    
    def verify(self, computed_result, context):
        """
        Check if computed result satisfies accounting constraints
        
        Returns:
            is_valid: Boolean
            violated_constraint: If invalid, which constraint was violated
        """
        # Extract relevant financial statement from context
        statement_type = self.identify_statement_type(context)
        
        # Get applicable constraints
        applicable_constraints = [c for c in self.constraints if self.is_applicable(c, statement_type)]
        
        for constraint in applicable_constraints:
            if not self.check_constraint(constraint, computed_result, context):
                return False, constraint
        
        return True, None
    
    def check_constraint(self, constraint, result, context):
        """
        Verify a specific accounting identity
        """
        lhs_expr, operator, rhs_expr = constraint
        
        # Parse expressions and evaluate
        lhs_value = self.evaluate_expression(lhs_expr, context)
        rhs_value = self.evaluate_expression(rhs_expr, context)
        
        if operator == "=":
            # Allow small floating point tolerance
            return abs(lhs_value - rhs_value) < 0.01
        elif operator == ">=":
            return lhs_value >= rhs_value
        elif operator == "<=":
            return lhs_value <= rhs_value
        
        return True
```

**Reasoning Router**:

Uses a lightweight classifier to determine reasoning mode:

```python
class ReasoningRouter:
    def __init__(self):
        self.classifier = self.load_classifier()
    
    def determine_mode(self, query, context):
        """
        Classify reasoning mode based on query and context characteristics
        
        Returns:
            "symbolic_only" | "neural_only" | "hybrid"
        """
        features = self.extract_features(query, context)
        
        # Features:
        # - Query contains explicit arithmetic keywords ("calculate", "sum", "total")
        # - Retrieved context contains only tables (vs only text vs mixed)
        # - Query complexity (simple vs multi-step)
        
        mode = self.classifier.predict(features)
        return mode
```

**Key Benefits of Fusion**:
1. **Precision**: Symbolic arithmetic eliminates floating-point errors in financial calculations
2. **Verification**: Constraint checking catches errors early
3. **Flexibility**: Neural component handles ambiguous queries
4. **Explainability**: Explicit calculation steps aid interpretability

### 2.7 Stage 6: Answer Generation with Attribution

**Objective**: Generate natural language answer with verifiable citations

**Attribution Requirements**:
1. **Cell-Level Attribution**: Each numerical claim → Specific table cell(s)
2. **Sentence-Level Attribution**: Each textual claim → Supporting sentence(s)
3. **Reasoning Chain**: Explicit steps from evidence to conclusion
4. **Confidence Scores**: Calibrated uncertainty estimates

**Generation Pipeline**:

```python
def generate_answer_with_attribution(query, retrieved_context, reasoning_trace):
    """
    Args:
        query: Original question
        retrieved_context: All retrieved elements
        reasoning_trace: Step-by-step reasoning from symbolic-neural fusion
    
    Returns:
        final_answer: Dict containing:
            - answer_text: Natural language answer
            - supporting_cells: List of (table_id, row, col, value)
            - supporting_sentences: List of (doc_id, paragraph_id, sentence)
            - reasoning_steps: List of computational/logical steps
            - confidence: Float in [0, 1]
    """
    # 1. Generate answer text
    generation_prompt = f"""
    Query: {query}
    
    Retrieved Evidence:
    {format_evidence(retrieved_context)}
    
    Reasoning Trace:
    {format_reasoning(reasoning_trace)}
    
    Generate a comprehensive answer that:
    1. Directly answers the question
    2. Cites specific evidence (mention table names, row labels)
    3. Explains any calculations performed
    4. Provides proper units and precision
    
    Format:
    Answer: [Your answer]
    Sources: [Specific cells/sentences used]
    Calculation: [If applicable, show computation]
    """
    
    answer_text = llm.generate(generation_prompt)
    
    # 2. Extract atomic claims from answer
    claims = extract_atomic_claims(answer_text)
    
    # 3. Map each claim to supporting evidence
    attribution_map = {}
    for claim in claims:
        supporting_evidence = find_supporting_evidence(claim, retrieved_context)
        attribution_map[claim] = supporting_evidence
    
    # 4. Compute confidence score
    confidence = compute_confidence(
        retrieval_quality=retrieved_context.avg_similarity,
        reasoning_validity=reasoning_trace.constraints_satisfied,
        attribution_coverage=len(attribution_map) / len(claims)
    )
    
    # 5. Compile final answer
    final_answer = {
        "answer_text": answer_text,
        "supporting_cells": extract_cells_from_attribution(attribution_map),
        "supporting_sentences": extract_sentences_from_attribution(attribution_map),
        "reasoning_steps": reasoning_trace.steps,
        "confidence": confidence
    }
    
    return final_answer
```

**Atomic Claim Extraction**:

Uses NLI model to decompose answer into verifiable atomic statements:

```
Answer: "The company's revenue grew 15% from $100M in 2023 to $115M in 2024."

Atomic Claims:
1. "The company's 2023 revenue was $100M"
2. "The company's 2024 revenue was $115M"
3. "Revenue growth was 15%"
4. "The growth calculation is (115-100)/100 = 0.15"
```

**Evidence Verification**:

For each atomic claim, verify entailment:

```python
def verify_claim_entailment(claim, evidence):
    """
    Use NLI model to check if evidence supports claim
    
    Returns:
        entailment_score: Float in [0, 1]
    """
    nli_model = load_nli_model("microsoft/deberta-v3-large-mnli")
    
    # For each evidence piece
    entailment_scores = []
    for evidence_piece in evidence:
        # NLI: Does evidence entail claim?
        input_text = f"{evidence_piece} [SEP] {claim}"
        output = nli_model(input_text)
        entailment_prob = output["entailment"]
        entailment_scores.append(entailment_prob)
    
    # Return maximum entailment (at least one piece strongly supports)
    return max(entailment_scores)
```

**Confidence Calibration**:

```python
def compute_confidence(retrieval_quality, reasoning_validity, attribution_coverage):
    """
    Combine multiple signals into calibrated confidence
    
    Args:
        retrieval_quality: Avg cosine similarity of retrieved docs (0-1)
        reasoning_validity: Boolean - constraints satisfied
        attribution_coverage: Fraction of claims with supporting evidence (0-1)
    
    Returns:
        confidence: Calibrated probability in [0, 1]
    """
    # Weighted combination
    base_confidence = (
        0.3 * retrieval_quality +
        0.4 * (1.0 if reasoning_validity else 0.0) +
        0.3 * attribution_coverage
    )
    
    # Apply calibration based on historical accuracy
    # (Learned from validation set)
    calibrated_confidence = apply_temperature_scaling(base_confidence)
    
    return calibrated_confidence
```

---

## 3. Comprehensive Evaluation Methodology

### 3.1 Evaluation Datasets

We will evaluate on **7 diverse datasets** covering different aspects of financial document understanding:

#### Dataset 1: **FinQA** (Primary Benchmark)

**Source**: Chen et al., EMNLP 2021  
**Size**: 8,281 questions over 2,516 financial reports  
**Task**: Numerical reasoning requiring multi-step arithmetic  

**Question Types**:
- Single-span extraction (15%)
- Multi-span extraction (25%)
- Arithmetic operations (35%)
- Multi-step reasoning (25%)

**Metrics**:
- **Exact Match (EM)**: Strict equality with gold answer
- **F1 Score**: Token-level overlap
- **Numerical Accuracy**: Within 1% of gold value
- **Program Accuracy**: For program-based approach, correctness of generated code

**Current SOTA**:
- GPT-4: 76% EM
- Human: 89-91% EM
- **Target**: 85%+ EM (close 60%+ of gap)

**Example Question**:
```
Context: [Table showing revenue: 2022=$100M, 2023=$115M, COGS: 2022=$60M, 2023=$70M]
Question: "What was the gross profit margin in 2023?"
Gold Answer: "39.13%"
Reasoning: (115-70)/115 * 100 = 39.13%
```

#### Dataset 2: **FinanceBench**

**Source**: Islam et al., 2023  
**Size**: 10,231 questions across 150 companies (S&P 500)  
**Task**: Complex financial document QA with evidence verification  

**Difficulty Levels**:
- Level 1 (Easy): Single-document, single-table (40%)
- Level 2 (Medium): Multi-table or calculation required (35%)
- Level 3 (Hard): Multi-document synthesis (25%)

**Special Features**:
- **Evidence Annotations**: Gold supporting sentences/cells
- **Numerical Focus**: 60% of questions require arithmetic
- **Cross-Reference**: 30% require resolving cross-references

**Metrics**:
- **EM Accuracy**: Exact match with gold answer
- **Evidence F1**: Overlap between predicted and gold evidence
- **Attribution Accuracy**: % of correct evidence citations

**Current Challenge**:
- RAGAS fails on 83.5% of numerical questions
- **Target**: 75%+ accuracy on numerical subset

**Example Question**:
```
Question: "What was JPMorgan's Tier 1 capital ratio in Q4 2023?"
Gold Answer: "15.8%"
Evidence: [10-K page 45, Capital Adequacy table, row "Tier 1 capital ratio", column "Dec 31, 2023"]
```

#### Dataset 3: **ConvFinQA**

**Source**: Chen et al., EMNLP 2022  
**Size**: 3,892 conversations with 14,115 questions  
**Task**: Conversational QA requiring multi-turn context  

**Characteristics**:
- **Sequential Reasoning**: Answer depends on previous QA pairs
- **Coreference**: Pronouns ("it", "this") refer to prior entities
- **Implicit Temporal**: "the previous year" requires tracking context

**Metrics**:
- **EM Accuracy**
- **Conversation Success Rate**: % of full conversations answered correctly
- **Context Utilization**: How well model uses prior turns

**Example Conversation**:
```
Turn 1: "What was Tesla's total revenue in 2023?"
Answer: "$96.8B"

Turn 2: "How does this compare to the previous year?"
Answer: "It increased by 18.8% from $81.5B in 2022"

Turn 3: "What drove this growth?"
Answer: [Requires retrieving MD&A section discussing growth drivers]
```

#### Dataset 4: **FinTable-X** (New Dataset - Our Contribution)

**Size**: 5,000 questions from 500 10-K filings (2020-2024)  
**Task**: Table-text integration with explicit attribution requirements  

**Question Categories**:
1. **Cross-Reference Resolution** (30%): "As mentioned in Note 5, what was the value?"
2. **Multi-Period Comparison** (25%): "Compare operating margin across 2022-2024"
3. **Hierarchical Aggregation** (20%): "What is total operating expense?" (requires summing multiple line items)
4. **Conditional Retrieval** (15%): "For segments with >$10M revenue, what are the operating margins?"
5. **Multi-Modal Synthesis** (10%): Requires both text and table

**Annotations**:
- Gold answer
- **Supporting cells**: Exact (table_id, row_idx, col_idx, value) coordinates
- **Supporting sentences**: (doc_id, paragraph_id, sentence_idx)
- **Reasoning chain**: Step-by-step human-annotated reasoning
- **Difficulty rating**: 1-5 scale

**Metrics**:
- **EM Accuracy**
- **Cell-Level Precision/Recall**: % of gold cells correctly identified
- **Sentence-Level Precision/Recall**: % of gold sentences correctly identified
- **Reasoning Chain Validity**: Human evaluation of generated reasoning
- **Attribution F1**: Harmonic mean of precision and recall for all evidence

**Target**: 80%+ EM on this challenging dataset

#### Dataset 5: **TAT-QA** (Tabular And Textual QA)

**Source**: Zhu et al., ACL 2021  
**Size**: 16,552 questions over 2,757 hybrid contexts  
**Task**: QA requiring understanding both tables and associated text  

**Operation Types**:
- Span extraction (35%)
- Arithmetic (30%)
- Counting (10%)
- Comparison (15%)
- Others (10%)

**Metrics**:
- **EM (Numerical)**: For questions with numerical answers
- **EM (Span)**: For questions with text span answers
- **F1 Score**: Token-level overlap

**Advantage**: Tests explicit table-text integration capability

#### Dataset 6: **MultiHiertt**

**Source**: Zhao et al., 2022  
**Size**: 10,440 QA pairs from hybrid hierarchical tables  
**Task**: Reasoning over tables with hierarchical structure  

**Challenges**:
- **Hierarchical Headers**: Multi-level column/row headers
- **Complex Operations**: Requires understanding table structure
- **Aggregation**: Many questions require summing across categories

**Metrics**:
- **EM Accuracy**
- **F1 Score**
- **Operation Accuracy**: % of correctly identified operations

**Why Important**: Financial statements often have hierarchical structure (e.g., revenue → product line → geography)

#### Dataset 7: **SEC-10K-QA** (New Test Set)

**Size**: 1,000 questions from recent 10-K filings (2024)  
**Purpose**: Out-of-distribution evaluation on truly unseen data  

**Question Generation**:
- GPT-4 generates diverse questions from 2024 filings not in training data
- Human expert review and answer verification
- Ensures model hasn't seen these documents during training

**Metrics**: Same as FinQA (EM, F1, Numerical Accuracy)

**Why Critical**: Tests generalization to new companies/time periods

### 3.2 Evaluation Metrics

We will report **15 comprehensive metrics** across different dimensions:

#### A. Answer Quality Metrics

**1. Exact Match (EM)**
```python
def exact_match(prediction, gold_answer):
    """
    Strict equality after normalization
    """
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(gold_answer)
    return 1.0 if pred_norm == gold_norm else 0.0

def normalize_answer(text):
    """
    Lowercase, remove punctuation, remove articles (a, an, the)
    """
    text = text.lower()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()
```

**Reported**: Accuracy across all questions (%)

**2. F1 Score (Token-Level)**
```python
def f1_score(prediction, gold_answer):
    """
    Token-level precision and recall
    """
    pred_tokens = set(prediction.lower().split())
    gold_tokens = set(gold_answer.lower().split())
    
    if len(gold_tokens) == 0:
        return 1.0 if len(pred_tokens) == 0 else 0.0
    
    common = pred_tokens & gold_tokens
    precision = len(common) / len(pred_tokens) if pred_tokens else 0.0
    recall = len(common) / len(gold_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1
```

**Reported**: Macro-average F1 across all questions

**3. Numerical Accuracy (Within Tolerance)**
```python
def numerical_accuracy(prediction, gold_answer, tolerance=0.01):
    """
    For numerical answers, check if within tolerance
    
    Args:
        tolerance: Relative error tolerance (1% by default)
    """
    try:
        pred_value = extract_number(prediction)
        gold_value = extract_number(gold_answer)
        
        if gold_value == 0:
            return 1.0 if pred_value == 0 else 0.0
        
        relative_error = abs(pred_value - gold_value) / abs(gold_value)
        return 1.0 if relative_error <= tolerance else 0.0
    except:
        # If not a numerical answer, fall back to EM
        return exact_match(prediction, gold_answer)
```

**Reported**: Accuracy on numerical questions (%)

**4. Scaled F1 (for numerical answers)**
```python
def scaled_f1(prediction, gold_answer):
    """
    For numerical answers, compute F1 based on relative error
    Proposed in FinQA paper
    """
    pred_value = extract_number(prediction)
    gold_value = extract_number(gold_answer)
    
    if gold_value == 0:
        return 1.0 if pred_value == 0 else 0.0
    
    relative_error = abs(pred_value - gold_value) / abs(gold_value)
    
    # F1 decreases linearly with relative error
    f1 = max(0.0, 1.0 - relative_error)
    return f1
```

#### B. Retrieval Quality Metrics

**5. Retrieval Precision @ k**
```python
def retrieval_precision_at_k(retrieved_cells, gold_cells, k):
    """
    What fraction of top-k retrieved cells are in gold set?
    
    Args:
        retrieved_cells: List of retrieved (table_id, row, col)
        gold_cells: Set of gold (table_id, row, col)
        k: Cutoff
    """
    top_k_retrieved = set(retrieved_cells[:k])
    correct = top_k_retrieved & gold_cells
    
    return len(correct) / k if k > 0 else 0.0
```

**Reported**: P@5, P@10, P@20

**6. Retrieval Recall @ k**
```python
def retrieval_recall_at_k(retrieved_cells, gold_cells, k):
    """
    What fraction of gold cells are in top-k retrieved?
    """
    top_k_retrieved = set(retrieved_cells[:k])
    correct = top_k_retrieved & gold_cells
    
    return len(correct) / len(gold_cells) if gold_cells else 0.0
```

**Reported**: R@5, R@10, R@20

**7. Mean Reciprocal Rank (MRR)**
```python
def mean_reciprocal_rank(retrieved_cells, gold_cells):
    """
    Average reciprocal rank of first gold cell
    """
    for rank, cell in enumerate(retrieved_cells, 1):
        if cell in gold_cells:
            return 1.0 / rank
    return 0.0
```

**Reported**: MRR across all questions

**8. NDCG@k (Normalized Discounted Cumulative Gain)**
```python
def ndcg_at_k(retrieved_cells, gold_cells, k):
    """
    Ranking quality metric accounting for position
    """
    dcg = 0.0
    for i, cell in enumerate(retrieved_cells[:k], 1):
        relevance = 1.0 if cell in gold_cells else 0.0
        dcg += relevance / np.log2(i + 1)
    
    # Ideal DCG (all gold cells retrieved first)
    ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(gold_cells), k)))
    
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0
```

**Reported**: NDCG@10

#### C. Attribution Quality Metrics

**9. Attribution Precision (Cell-Level)**
```python
def attribution_precision_cells(predicted_cells, gold_cells):
    """
    Of predicted supporting cells, how many are correct?
    """
    if not predicted_cells:
        return 0.0
    
    correct = set(predicted_cells) & set(gold_cells)
    return len(correct) / len(predicted_cells)
```

**10. Attribution Recall (Cell-Level)**
```python
def attribution_recall_cells(predicted_cells, gold_cells):
    """
    Of gold supporting cells, how many were predicted?
    """
    if not gold_cells:
        return 1.0  # Vacuously true if no gold cells
    
    correct = set(predicted_cells) & set(gold_cells)
    return len(correct) / len(gold_cells)
```

**11. Attribution F1 (Cell-Level)**
```python
def attribution_f1_cells(predicted_cells, gold_cells):
    """
    Harmonic mean of precision and recall
    """
    precision = attribution_precision_cells(predicted_cells, gold_cells)
    recall = attribution_recall_cells(predicted_cells, gold_cells)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)
```

**Reported**: F1 on FinTable-X which has cell-level annotations

**12. Attribution F1 (Sentence-Level)**
```python
def attribution_f1_sentences(predicted_sentences, gold_sentences):
    """
    Same as cell-level but for text sentences
    """
    # Similar computation as above
    ...
```

#### D. Reasoning Quality Metrics

**13. Reasoning Chain Validity (Human Evaluation)**

**Process**:
- Sample 200 questions from test set
- Human annotators (financial domain experts) review generated reasoning chains
- Rate on 5-point Likert scale:
  - 1: Completely incorrect/illogical
  - 2: Mostly incorrect with some valid steps
  - 3: Partially correct but missing key steps
  - 4: Mostly correct with minor issues
  - 5: Completely correct and logical

**Metrics**:
- Average rating (1-5)
- % of ratings ≥4 (acceptable quality)
- Inter-annotator agreement (Krippendorff's α)

**14. Operation Accuracy**
```python
def operation_accuracy(predicted_op, gold_op):
    """
    For questions requiring arithmetic, did model select correct operation?
    
    Operations: {add, subtract, multiply, divide, percentage, ratio, count}
    """
    return 1.0 if predicted_op == gold_op else 0.0
```

**Reported**: Accuracy on subset of questions with explicit operation annotations

#### E. Confidence Calibration Metrics

**15. Expected Calibration Error (ECE)**
```python
def expected_calibration_error(predictions, confidences, n_bins=10):
    """
    Measure calibration of confidence scores
    
    Args:
        predictions: List of (prediction, gold_answer) pairs
        confidences: List of confidence scores [0, 1]
        n_bins: Number of bins for histogram
    
    Returns:
        ECE: Lower is better (0 = perfect calibration)
    """
    # Bin predictions by confidence
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find predictions in this bin
        in_bin = [(p, g) for (p, g), c in zip(predictions, confidences)
                  if bin_lower <= c < bin_upper]
        
        if len(in_bin) == 0:
            continue
        
        # Compute accuracy in bin
        accuracy_in_bin = np.mean([exact_match(p, g) for p, g in in_bin])
        
        # Average confidence in bin
        avg_confidence_in_bin = np.mean([c for (p, g), c in zip(predictions, confidences)
                                         if bin_lower <= c < bin_upper])
        
        # ECE contribution
        ece += (len(in_bin) / len(predictions)) * abs(accuracy_in_bin - avg_confidence_in_bin)
    
    return ece
```

**Reported**: ECE across test set

### 3.3 Baseline Comparisons

We will compare against **8 strong baselines**:

#### Baseline 1: **Vanilla RAG**
- **Retrieval**: Dense retrieval (E5-Mistral-7B embeddings)
- **Generation**: GPT-4o
- **Description**: Flatten tables to text, standard retrieve-read pattern
- **Purpose**: Establish lower bound

#### Baseline 2: **ColPali**
- **Method**: Vision-based document understanding
- **Retrieval**: Late interaction over document screenshots
- **Generation**: Qwen2-VL or GPT-4o with vision
- **Purpose**: SOTA vision-based approach

#### Baseline 3: **Table-BERT + RAG**
- **Method**: Specialized table representation learning
- **Model**: TaPas or TAPEX for table encoding
- **Generation**: GPT-4o
- **Purpose**: SOTA table-focused approach

#### Baseline 4: **GPT-4o with Code Interpreter**
- **Method**: LLM generates Python code to compute answer
- **Tools**: Can use pandas, numpy for table processing
- **Purpose**: Current best system on FinQA (76% EM)

#### Baseline 5: **GraphRAG (Microsoft)**
- **Method**: Knowledge graph based retrieval
- **Adaptation**: Build KG from financial documents
- **Generation**: GPT-4o
- **Purpose**: Test graph-based methods (not multimodal specialized)

#### Baseline 6: **Self-RAG**
- **Method**: Adaptive retrieval with self-critique
- **Model**: Llama-3.1-70B fine-tuned with reflection tokens
- **Purpose**: Test agentic baseline

#### Baseline 7: **RAFT (Fine-tuned)**
- **Method**: Fine-tune Llama-3.1-70B on financial QA with RAG
- **Training**: Include oracle and distractor documents
- **Purpose**: Test fine-tuning approach

#### Baseline 8: **Ensemble: GPT-4o + Claude-3.5-Sonnet**
- **Method**: Generate answers with both LLMs, select based on confidence
- **Purpose**: Upper bound with multiple strong models

### 3.4 Ablation Studies

To understand contribution of each component, we will conduct **systematic ablations**:

#### Ablation 1: Architecture Components

| Configuration | Description | Purpose |
|--------------|-------------|---------|
| Text-Only | Remove all tables, use only narrative text | Test multimodal necessity |
| Table-Only | Remove narrative, use only tables | Test if tables sufficient |
| No Graph | Skip TTGNN, use flat retrieval | Test graph benefit |
| No Hierarchy | Flat retrieval (skip Level 1, 2) | Test hierarchical attention benefit |
| Full Model | Complete HierFinRAG | - |

**Metrics**: Compare EM accuracy on FinQA and FinanceBench

#### Ablation 2: Reasoning Modes

| Configuration | Description |
|--------------|-------------|
| Neural-Only | Skip symbolic calculator, pure LLM generation |
| Symbolic-Only | Rule-based arithmetic, no LLM flexibility |
| No Constraint Checking | Skip accounting constraint verification |
| Full Hybrid | Complete symbolic-neural fusion |

**Metrics**: Compare numerical accuracy and operation accuracy

#### Ablation 3: Retrieval Strategies

| Configuration | Description |
|--------------|-------------|
| Dense-Only | Only semantic similarity (no BM25) |
| Sparse-Only | Only BM25 (no dense embeddings) |
| No Reranking | Skip RRF fusion |
| No Intent Classification | Single retrieval strategy for all queries |
| Full Pipeline | Complete hierarchical retrieval |

**Metrics**: Compare retrieval precision/recall and NDCG

#### Ablation 4: Attribution Components

| Configuration | Description |
|--------------|-------------|
| No Attribution | Generate answer without evidence tracking |
| Cell Attribution Only | Track table cells but not text sentences |
| Sentence Attribution Only | Track text but not cells |
| Full Attribution | Complete cell + sentence + reasoning chain |

**Metrics**: Attribution F1, reasoning chain validity

### 3.5 Error Analysis

**Systematic Error Categorization**:

We will manually analyze 200 error cases across categories:

1. **Retrieval Errors** (Expected ~40% of errors)
   - Gold evidence not retrieved
   - Retrieved evidence insufficient
   - Irrelevant evidence ranked highly

2. **Reasoning Errors** (Expected ~30% of errors)
   - Incorrect operation selection (add vs multiply)
   - Arithmetic mistakes (despite calculator)
   - Constraint violation (violated accounting identity)
   - Multi-hop failure (error propagation)

3. **Generation Errors** (Expected ~20% of errors)
   - Correct reasoning but wrong final answer format
   - Hallucinated values not in context
   - Misinterpretation of units (millions vs thousands)

4. **Cross-Reference Resolution Errors** (Expected ~10% of errors)
   - Failed to follow "See Note X" references
   - Linked to wrong table/section
   - Ambiguous reference handling

**Deliverable**: Detailed error analysis document with representative examples

### 3.6 Generalization Tests

#### Test 1: **Cross-Company Generalization**

**Setup**:
- Train on companies A, B, C
- Test on companies X, Y, Z (different industries)

**Metrics**: Measure performance drop on unseen companies

#### Test 2: **Temporal Generalization**

**Setup**:
- Train on 2020-2023 filings
- Test on 2024 filings

**Metrics**: Measure recency bias and adaptation

#### Test 3: **Cross-Industry Generalization**

**Setup**:
- Train on Tech sector
- Test on Finance, Healthcare, Retail sectors

**Metrics**: Identify industry-specific challenges

#### Test 4: **Adversarial Robustness**

**Setup**:
- Add distractor tables with similar headers
- Inject confusing cross-references
- Add numerical values with wrong units

**Metrics**: Measure robustness to adversarial perturbations

### 3.7 Efficiency Metrics

**Latency**:
- **End-to-End**: Total time from query to answer
- **Retrieval**: Time for hierarchical retrieval
- **Graph Construction**: Offline parsing time per document
- **Target**: <10 seconds per query

**Memory**:
- **Graph Size**: Average nodes/edges per 10-K
- **Index Size**: Total vector database size
- **Peak GPU Memory**: During inference

**Cost**:
- **Per-Query Cost**: Embedding + LLM API calls
- **Total Cost**: For full evaluation on all datasets

### 3.8 Statistical Significance Testing

**Method**: Bootstrap resampling with 1,000 iterations

```python
def bootstrap_significance_test(model_a_results, model_b_results, n_bootstrap=1000):
    """
    Test if Model A significantly outperforms Model B
    
    Returns:
        p_value: Probability that difference is due to chance
        confidence_interval: 95% CI for performance difference
    """
    n = len(model_a_results)
    differences = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        
        sample_a = [model_a_results[i] for i in indices]
        sample_b = [model_b_results[i] for i in indices]
        
        diff = np.mean(sample_a) - np.mean(sample_b)
        differences.append(diff)
    
    # Compute p-value (two-tailed)
    p_value = np.mean(np.array(differences) <= 0) * 2
    
    # 95% confidence interval
    ci_lower = np.percentile(differences, 2.5)
    ci_upper = np.percentile(differences, 97.5)
    
    return p_value, (ci_lower, ci_upper)
```

**Reporting**: All comparisons will include p-values and state significance at α=0.05 level

### 3.9 Human Evaluation

**Setup**: Recruit 5 financial analysts with 5+ years experience

**Tasks**:

1. **Ranking Task**: Present answer from 3 systems (HierFinRAG, GPT-4o, ColPali)
   - Rank from best to worst
   - 100 questions × 5 annotators = 500 judgments

2. **Absolute Rating**: Rate HierFinRAG answers on 1-5 scale
   - Correctness (1-5)
   - Completeness (1-5)
   - Evidence quality (1-5)
   - Explanation clarity (1-5)
   - 200 questions × 5 annotators = 1,000 ratings

3. **Error Identification**: Identify errors in answers
   - What went wrong?
   - How to fix it?
   - 50 error cases × 5 annotators

**Inter-Annotator Agreement**: Compute Fleiss' κ and report

**Compensation**: $50/hour, estimated 10 hours per annotator

---

## 4. Expected Results and Success Criteria

### 4.1 Primary Success Criteria

**Minimum Viable Results** (Must achieve):

1. **FinQA EM**: 82%+ (vs GPT-4o: 76%, Gap closure: 40%+)
2. **FinanceBench Numerical**: 70%+ (vs RAGAS failure on 83.5%)
3. **FinTable-X EM**: 75%+ (new benchmark, establish SOTA)
4. **Attribution F1**: 80%+ on FinTable-X
5. **Statistical Significance**: p<0.05 vs all baselines on primary metrics

**Target Results** (Aim to achieve):

1. **FinQA EM**: 85%+ (Gap closure: 60%+)
2. **FinanceBench Numerical**: 75%+
3. **FinTable-X EM**: 80%+
4. **Attribution F1**: 85%+
5. **Human Preference**: Win >60% of pairwise comparisons vs GPT-4o

**Stretch Results** (Aspirational):

1. **FinQA EM**: 87%+ (Gap closure: 73%+)
2. **FinanceBench Numerical**: 80%+
3. **FinTable-X EM**: 85%+
4. **Attribution F1**: 90%+
5. **Human Preference**: Win >70% vs GPT-4o

### 4.2 Component-Level Expectations

**Retrieval Quality**:
- Precision@10: 75%+ (at least 7-8 of top-10 retrieved cells are relevant)
- Recall@20: 90%+ (capture 90%+ of gold cells within top-20)
- NDCG@10: 0.85+ (good ranking quality)

**Reasoning Accuracy**:
- Operation Accuracy: 95%+ (correct arithmetic operation selection)
- Constraint Satisfaction: 98%+ (satisfy accounting identities)
- Reasoning Chain Validity: 4.0+ average rating (human evaluation)

**Confidence Calibration**:
- ECE: <0.1 (well-calibrated confidence scores)
- High-confidence predictions: 95%+ accuracy when confidence >0.9
- Low-confidence flagging: Correctly abstain on difficult questions

### 4.3 Ablation Study Expectations

**Multimodal vs Unimodal**:
- Full multimodal should outperform text-only by 15%+ on FinQA
- Full multimodal should outperform table-only by 10%+ on questions requiring context

**Hierarchical vs Flat Retrieval**:
- Hierarchical attention should improve precision by 10-15% (fewer irrelevant retrievals)
- Speed improvement: 2-3× faster due to reduced search space

**Symbolic-Neural Fusion**:
- Hybrid reasoning should outperform neural-only by 8-12% on numerical questions
- Constraint checking should catch and correct 10-15% of errors

**Graph vs No-Graph**:
- TTGNN should improve over flat embeddings by 5-8% (better context understanding)
- Particularly strong on cross-reference questions (20%+ improvement)

---

## 5. Risk Mitigation and Contingency Plans

### Risk 1: **Dataset Annotation Quality**

**Risk**: Crowdsourced annotations for FinTable-X may be inconsistent or incorrect

**Mitigation**:
- **Multi-Annotator Agreement**: Require 3 annotators per question, use majority vote
- **Expert Review**: Financial domain expert reviews 20% of annotations
- **Quality Control**: Reject annotations with <70% agreement
- **Gold Standard**: Create 100-question gold set with certified correct answers for calibration

**Contingency**: If quality insufficient, reduce dataset size and use only expert-annotated subset

### Risk 2: **TTGNN Training Instability**

**Risk**: Graph neural networks can be difficult to train, especially with heterogeneous node/edge types

**Mitigation**:
- **Pre-training**: Pre-train on synthetic data (generated table-text pairs)
- **Curriculum Learning**: Start with simple graphs (small docs), progress to complex
- **Careful Initialization**: Use pre-trained text embeddings, not random initialization
- **Gradient Clipping**: Prevent exploding gradients
- **Extensive Hyperparameter Tuning**: Grid search over learning rate, hidden dims, num layers

**Contingency**: If GNN doesn't converge, fall back to simpler graph pooling or attention mechanisms

### Risk 3: **Symbolic Calculator Brittleness**

**Risk**: Rule-based calculator may fail on edge cases or unusual query formulations

**Mitigation**:
- **Extensive Unit Testing**: Test on 1000+ synthetic arithmetic problems
- **Fallback Mechanism**: If symbolic fails, fall back to neural-only generation
- **Learned Confidence Threshold**: Only invoke symbolic when high confidence in operation detection
- **Error Recovery**: Catch exceptions and retry with alternative interpretation

**Contingency**: Make symbolic component optional, report results with/without it

### Risk 4: **Performance Gains Insufficient**

**Risk**: Despite all innovations, may not achieve 60%+ gap closure on FinQA

**Mitigation**:
- **Iterative Refinement**: Build incrementally, test each component
- **Error Analysis Early**: After initial results, deep dive into failure modes
- **Ensemble Methods**: If single model insufficient, ensemble multiple configurations
- **Larger Models**: If budget allows, test GPT-4o instead of GPT-3.5 for generation
- **Fine-tuning**: If inference-time methods insufficient, fine-tune generator on financial QA

**Contingency**: Even with <60% gap closure, any significant improvement (>5%) over baselines is publishable if accompanied by thorough analysis

### Risk 5: **Computational Cost Prohibitive**

**Risk**: TTGNN + hierarchical retrieval may be too slow for practical use

**Mitigation**:
- **Graph Sampling**: Use NeighborSampler to avoid full graph processing
- **Caching**: Cache frequently accessed embeddings and subgraphs
- **Batch Processing**: Process multiple queries in parallel
- **Optimized Implementation**: Use PyTorch Geometric for efficient GNN ops
- **Pruning**: Remove low-weight edges to reduce graph density

**Contingency**: Report both accuracy and efficiency, acknowledge accuracy-speed tradeoff

### Risk 6: **Baseline Performance Higher Than Expected**

**Risk**: GPT-4o or other baselines may perform better than reported in literature

**Mitigation**:
- **Re-run Baselines**: Implement baselines ourselves with best practices
- **Fair Comparison**: Use same retrieval corpus and experimental setup
- **Multiple Runs**: Average over 3 runs to account for LLM variance
- **Ablation Focus**: Even if absolute gains small, demonstrate which components help

**Contingency**: If baselines very strong, pivot to emphasizing interpretability, efficiency, or specialized domains where baselines fail

---

## 6. Open-Source Release Plan

### Components to Release:

1. **FinTable-X Dataset**
   - 5,000 annotated questions with evidence
   - Released under CC BY 4.0 license
   - Hosted on Hugging Face Datasets

2. **HierFinRAG Code**
   - Complete implementation in PyTorch
   - Docker container for reproducibility
   - Released under MIT license
   - GitHub repository with documentation

3. **Pre-trained Models**
   - TTGNN checkpoint
   - Financial Intent Classifier
   - Hosted on Hugging Face Model Hub

4. **Evaluation Suite**
   - Scripts to reproduce all results
   - Baseline implementations
   - Jupyter notebooks with examples

5. **Interactive Demo**
   - Streamlit or Gradio web interface
   - Upload 10-K, ask questions, see attributions
   - Deployed on Hugging Face Spaces

### Documentation:

- **README**: Quick start guide
- **Paper**: Detailed methodology
- **API Documentation**: Function-level docs
- **Tutorial**: Step-by-step walkthrough
- **FAQ**: Common issues and solutions

---

## 7. Publication Strategy

### Target Venue: MDPI Q2 Journals

**Primary Targets** (in order of preference):

1. **MDPI Applied Sciences** - Section: "Computing and Artificial Intelligence"
   - IF: ~2.5, Q2 in Computer Science (Applications)
   - Scope: Applied AI systems
   - Typical length: 8,000-12,000 words
   - Review time: ~4-6 weeks

2. **MDPI Information** - Section: "Information Systems and Applications"
   - IF: ~3.1, Q2 in Information Systems
   - Scope: Information systems, data management
   - Typical length: 8,000-10,000 words
   - Review time: ~4-6 weeks

3. **MDPI Electronics** - Section: "Artificial Intelligence"
   - IF: ~2.6, Q2 in Engineering (Electrical)
   - Scope: AI hardware and software
   - Typical length: 7,000-10,000 words

### Paper Structure (10,000 words):

1. **Abstract** (250 words)
2. **Introduction** (1,500 words)
   - Problem motivation
   - Limitations of current approaches
   - Key contributions

3. **Related Work** (1,500 words)
   - RAG systems
   - Multimodal learning
   - Financial NLP
   - Clear positioning vs prior work

4. **Methodology** (3,000 words)
   - Architecture overview
   - TTGNN details
   - Hierarchical retrieval
   - Symbolic-neural fusion
   - Implementation details

5. **Experimental Setup** (1,000 words)
   - Datasets (7 datasets described)
   - Evaluation metrics (15 metrics)
   - Baseline systems
   - Hyperparameters

6. **Results and Analysis** (2,000 words)
   - Main results tables
   - Ablation studies
   - Error analysis
   - Statistical significance tests

7. **Discussion** (500 words)
   - Key insights
   - Limitations
   - Broader impact
   - Future work

8. **Conclusion** (250 words)

### Key Contributions (for Abstract):

1. **HierFinRAG Architecture**: First hierarchical multimodal RAG specifically designed for financial documents
2. **TTGNN**: Novel table-text graph neural network for unified representation
3. **Symbolic-Neural Fusion**: Combines neural flexibility with symbolic arithmetic precision
4. **FinTable-X Dataset**: 5,000 annotated questions with cell-level attribution
5. **Comprehensive Evaluation**: 7 datasets, 15 metrics, 8 baselines
6. **SOTA Performance**: Close 60%+ of human-AI gap on FinQA, 75%+ on numerical FinanceBench

### Differentiation from Prior Work:

**vs ColPali**: Adds symbolic reasoning and hierarchical retrieval (not just visual encoding)

**vs GraphRAG**: Specialized for financial documents with table-text integration and accounting constraints

**vs RAFT**: Inference-time architecture innovations (not just fine-tuning method)

**vs MMed-RAG**: Financial domain with regulatory compliance focus, different attribution requirements

**vs Self-RAG**: Multimodal reasoning, not just text; hierarchical attention; symbolic component

---

## 8. Conclusion

HierFinRAG represents a comprehensive solution to financial document understanding through:

1. **Unified Multimodal Architecture**: TTGNN jointly models tables, text, and cross-references
2. **Hierarchical Efficiency**: Three-level retrieval reduces search space while maintaining accuracy
3. **Hybrid Reasoning**: Symbolic-neural fusion balances flexibility and precision
4. **Verifiable Attribution**: Cell-level and sentence-level evidence with reasoning chains
5. **Rigorous Evaluation**: 7 datasets, 15 metrics, 8 baselines, statistical significance testing

**Expected Impact**:
- Close 60-80% of human-AI performance gap on financial document QA
- Enable automated regulatory filing analysis (multi-billion dollar opportunity)
- Provide open-source tools and datasets for research community
- Demonstrate path to EU AI Act-compliant explainable RAG systems

**Success Criteria**:
- FinQA EM: 85%+ (vs 76% GPT-4o, 89-91% human)
- FinanceBench Numerical: 75%+ (vs RAGAS 83.5% failure rate)
- FinTable-X: 80%+ EM (establish SOTA on new benchmark)
- Attribution F1: 85%+ (verifiable evidence chains)
- Publication in MDPI Q2 journal with open-source release

This research will advance the state-of-the-art in financial AI while providing practical tools for industry adoption and establishing foundations for trustworthy, explainable RAG systems in high-stakes financial applications.
