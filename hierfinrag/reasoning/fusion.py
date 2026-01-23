import torch
import torch.nn as nn
import re
from typing import Dict, Any, List, Optional, Tuple, Literal

class SymbolicCalculator:
    def __init__(self, precision=2):
        self.precision = precision
        
    def compute(self, operation: str, values: Dict[str, float]) -> float:
        """
        Executes symbolic arithmetic operations.
        """
        try:
            if operation == "add":
                return round(sum(values.values()), self.precision)
            
            elif operation == "subtract":
                vals = list(values.values())
                if len(vals) != 2:
                    raise ValueError("Subtraction requires exactly 2 values")
                return round(vals[0] - vals[1], self.precision)
                
            elif operation == "divide" or operation == "ratio":
                num = values.get("numerator", list(values.values())[0])
                den = values.get("denominator", list(values.values())[1])
                if den == 0:
                    raise ValueError("Division by zero")
                return round(num / den, self.precision)
            
            elif operation == "percentage_change":
                old = values.get("old_value", values.get("previous"))
                new = values.get("new_value", values.get("current"))
                if old == 0: raise ValueError("Percentage change from zero")
                return round(((new - old) / old) * 100, self.precision)
                
            else:
                raise ValueError(f"Unknown operation: {operation}")
        except Exception as e:
            print(f"Symbolic Calculation Error: {e}")
            return 0.0

class ReasoningRouter:
    """
    Determines whether to use Neural, Symbolic, or Hybrid reasoning.
    Currently uses rule-based heuristics to simulate a trained classifier.
    Returns probability distribution per mode.
    """
    def __init__(self):
        # Keywords that strongly suggest symbolic reasoning
        self.symbolic_keywords = [
            "calculate", "sum", "total", "difference", "growth", "increase", 
            "decrease", "margin", "ratio", "percentage", "average"
        ]
        
    def predict(self, query: str, context_types: List[str]) -> Dict[str, float]:
        """
        Predict reasoning mode probabilities.
        Args:
            query: User's question
            context_types: List of types of retrieved nodes (e.g. ['Table', 'Text'])
        """
        query_lower = query.lower()
        
        # Base probabilities
        probs = {"neural": 0.3, "symbolic": 0.3, "hybrid": 0.4}
        
        # 1. Keyword analysis
        symbolic_score = sum(1 for k in self.symbolic_keywords if k in query_lower)
        if symbolic_score > 0:
            probs["symbolic"] += 0.4
            probs["hybrid"] += 0.2
            probs["neural"] -= 0.2
            
        # 2. Context analysis
        has_table = "Table" in context_types or "Cell" in context_types
        if has_table:
            probs["symbolic"] += 0.2
            probs["neural"] -= 0.1
        else:
            probs["neural"] += 0.3
            probs["symbolic"] -= 0.2
            
        # Normalize
        total = sum(probs.values())
        return {k: v / total for k, v in probs.items()}

    def determine_mode(self, query: str, context_types: List[str]) -> str:
        probs = self.predict(query, context_types)
        return max(probs, key=probs.get)

class SymbolicNeuralFusion(nn.Module):
    def __init__(self, llm_client=None):
        super().__init__()
        self.llm = llm_client # Object with .generate(prompt)
        self.calculator = SymbolicCalculator()
        self.router = ReasoningRouter()
        
    def forward(self, query: str, retrieved_nodes: List[Any]):
        """
        Main fusion logic.
        """
        # Extract context types
        context_types = [] 
        # Assuming retrieved_nodes are objects with a 'type' attribute or similar
        # For this prototype we'll assume they are simple dicts or objects
        for node in retrieved_nodes:
            # Simplification for demo
            if hasattr(node, 'type'): context_types.append(node.type)
            elif isinstance(node, dict) and 'type' in node: context_types.append(node['type'])
            else: context_types.append('Text')
            
        # 1. Routing
        mode_probs = self.router.predict(query, context_types)
        best_mode = max(mode_probs, key=mode_probs.get)
        
        print(f"  [Router] Mode: {best_mode.upper()} (Probs: {mode_probs})")
        
        if best_mode == "neural":
            # Pure LLM generation
            return f"[Neural Response] Generated answer based on text context."
            
        elif best_mode == "symbolic":
            # Just calc (simulated extraction)
            val1 = 100.0 # Mock extracted value
            val2 = 120.0
            result = self.calculator.compute("percentage_change", {"old_value": val1, "new_value": val2})
            return f"[Symbolic Response] Calculated Result: {result}%"
            
        else: # Hybrid
            # 1. LLM extracts plan (Mock)
            # 2. Calculator executes
            # 3. LLM synthesizes
            plan = {"op": "percentage_change", "vars": {"old": 100, "new": 120}}
            result = self.calculator.compute(plan["op"], {"old_value": plan["vars"]["old"], "new_value": plan["vars"]["new"]})
            return f"[Hybrid Response] Revenue grew by {result}% from 2022 to 2023."
