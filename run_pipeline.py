import os
import sys
import json
import torch
from hierfinrag.parsing.json_parser import JSONParser
from hierfinrag.graph.builder import GraphBuilder
from hierfinrag.graph.ttgnn import TTGNN
from hierfinrag.reasoning.fusion import SymbolicNeuralFusion

def main():
    print("Starting HierFinRAG Phase 1 Pipeline...")
    
    # 1. Create Mock Data
    sample_data = {
        "id": "doc_demo",
        "title": "Annual Report 2023",
        "sections": [
            {
                "id": "s1", "title": "Financial Highlights", "level": 1, 
                "content_ids": ["p1", "t1"]
            }
        ],
        "paragraphs": [
            {"id": "p1", "text": "Net income increased by 15% due to strong sales growth as shown in Table 1.", "section_id": "s1"}
        ],
        "tables": [
            {
                "id": "t1", 
                "caption": "Consolidated Statement of Income",
                "col_headers": ["2022", "2023"],
                "row_headers": ["Revenue", "Net Income"],
                "cells": [
                    {"row": 0, "col": 0, "value": "100M", "is_header": False},
                    {"row": 0, "col": 1, "value": "120M", "is_header": False},
                    {"row": 1, "col": 0, "value": "10M", "is_header": False},
                    {"row": 1, "col": 1, "value": "11.5M", "is_header": False} # 15% of 10M is 1.5M increase -> 11.5M
                ]
            }
        ]
    }
    
    # Save mock data
    os.makedirs("data", exist_ok=True)
    mock_file = "data/mock_parsed.json"
    with open(mock_file, 'w') as f:
        json.dump(sample_data, f, indent=2)
    print(f"Created mock data at {mock_file}")
    
    # 2. Parse Data
    print("Parsing document...")
    parser = JSONParser()
    doc = parser.parse(mock_file)
    print(f"Parsed Document: {doc.title}")
    print(f" - {len(doc.sections)} Sections")
    print(f" - {len(doc.paragraphs)} Paragraphs")
    print(f" - {len(doc.tables)} Tables")
    
    # 3. Build Graph
    print("Constructing Graph...")
    builder = GraphBuilder(embedding_dim=128) # Smaller dim for demo
    graph = builder.build_graph(doc)
    
    print("\n--- Graph Statistics ---")
    print(f"Node Features: {graph.x.shape}")
    print(f"Edge Index:    {graph.edge_index.shape}")
    print(f"Edge Attr:     {graph.edge_attr.shape}")
    print(f"Node Types:    {graph.node_types.unique(return_counts=True)}")
    
    
    # 4. Initialize & Run TTGNN
    print("\nInitializing TTGNN...")
    # dimensions: input 128 (matches builder), hidden 64
    model = TTGNN(input_dim=128, hidden_dim=64, num_layers=2)
    model.eval()
    
    with torch.no_grad():
        out_embeddings = model(
            graph.x, 
            graph.edge_index, 
            graph.edge_attr, 
            graph.node_types
        )
    
    print(f"TTGNN Output Shape: {out_embeddings.shape}")
    
    # 5. Run Symbolic-Neural Fusion
    print("\nTesting Symbolic-Neural Fusion...")
    fusion_engine = SymbolicNeuralFusion(llm_client=None) # Mock LLM
    
    # Test Query 1: Symbolic
    q1 = "Calculate the percentage growth in net income from 2022 to 2023"
    print(f"Query: {q1}")
    ans1 = fusion_engine(q1, [{"type": "Table"}, {"type": "Cell"}]) # Mock retrieval context
    print(f"Result: {ans1}")
    
    # Test Query 2: Neural
    q2 = "Summarize the financial highlights"
    print(f"Query: {q2}")
    ans2 = fusion_engine(q2, [{"type": "Section"}, {"type": "Paragraph"}])
    print(f"Result: {ans2}")

    print("\nSUCCESS: Pipeline integration test passed.")


if __name__ == "__main__":
    main()
