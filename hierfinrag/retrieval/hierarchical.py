import torch
import numpy as np
import re
from typing import List, Any

# Placeholders for external models/utilities
class MockEmbedModel:
    def encode(self, text):
        # Return random embedding or mock
        if isinstance(text, list):
            return np.random.rand(len(text), 1024)
        return np.random.rand(1, 1024)

embed_model = MockEmbedModel()

def cosine_similarity(a, b):
    # Mock cosine similarity
    # a: [n, d], b: [m, d]
    if isinstance(a, torch.Tensor):
        a = a.detach().numpy()
    if isinstance(b, torch.Tensor):
        b = b.detach().numpy()
    # Simplified
    return np.dot(a, b.T)

def reciprocal_rank_fusion(scores_list, k=60):
    # Simplified RRF for 2 lists of scores (assuming they are ranks or we convert to ranks)
    # The snippet implies we have `bm25_scores` and `dense_scores` which are raw scores.
    # We should convert them to ranks first.
    # Placeholder implementation
    return np.array(scores_list[0]) + np.array(scores_list[1]) # Very naive

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
    if isinstance(query_emb, np.ndarray):
        query_emb = torch.tensor(query_emb)
    if isinstance(section_embeddings, np.ndarray):
        section_embeddings = torch.tensor(section_embeddings)

    # Manual cosine sim if not normalized
    # Assuming normalized for dot product
    similarities = torch.mm(query_emb, section_embeddings.t())
    
    # Get top-k sections
    top_k_indices = torch.topk(similarities, k=k).indices
    
    return top_k_indices

def level2_retrieval(query, section_nodes, bm25_retriever, k=10):
    """
    Args:
        query: Query text
        section_nodes: List of [Paragraph, Table] nodes in selected sections
        k: Number of elements to retrieve
    
    Returns:
        top_k_elements: Most relevant paragraphs/tables
    """
    # Sparse retrieval (BM25)
    # bm25_scores = bm25_retriever.score(query, section_nodes)
    bm25_scores = np.random.rand(len(section_nodes)) # Placeholder
    
    # Dense retrieval (embedding similarity)
    # query_emb = embed_model.encode(query)
    # dense_scores = cosine_similarity(query_emb, [n.embedding for n in section_nodes])
    dense_scores = np.random.rand(len(section_nodes)) # Placeholder
    
    # Reciprocal Rank Fusion
    # combined_scores = reciprocal_rank_fusion(bm25_scores, dense_scores)
    combined_scores = bm25_scores + dense_scores # Placeholder
    
    # Get top-k
    top_k_indices = np.argsort(combined_scores)[-k:][::-1]
    
    return [section_nodes[i] for i in top_k_indices]

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
        # assuming row.cells is list of objects with .value
        # and row.header is string.
        cell_values = [getattr(c, 'value', str(c)) for c in row.cells]
        row_text = str(row.header) + " " + " ".join(cell_values)
        row_representations.append(row_text)
    
    # Embed and compute similarity
    query_emb = embed_model.encode(query)
    row_embs = embed_model.encode(row_representations)
    similarities = cosine_similarity(query_emb, row_embs)
    
    top_k_indices = np.argsort(similarities.flatten())[-top_k:][::-1]
    return [table.rows[i] for i in top_k_indices]

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
            # Assuming table.columns is list of objects with .header
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

def extract_answer_cells(relevant_rows, relevant_columns, table):
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
