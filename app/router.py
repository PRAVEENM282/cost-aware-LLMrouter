"""
Semantic Router for LLM Gateway.
Uses SentenceTransformers to compute embeddings and cosine similarity against category anchors.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_ANCHORS: Dict[str, List[str]] = {
    "arithmetic": [
        "Calculate this math problem",
        "What is the product of two numbers",
        "Add these numbers together",
        "Subtract and find the difference",
        "Multiply these integers",
        "Divide this number by another",
        "Compute mathematical arithmetic expression",
        "What is 10 + 10",
        "Calculate 15 + 27",
        "What is 24 * 193",
        "What is 100 - 45",
        "Compute 1200 / 30",
        "Evaluate 99 * 3",
        "What is 144 / 12",
        "What is 125 * 8",
        "What is 512 / 8",
        "Calculate 87 - 39",
        "What is 16 * 16",
        "Solve numeric arithmetic calculation",
    ],
    "revenue": [
        "What was our quarterly revenue",
        "Show revenue for quarter",
        "What is Q3 revenue",
        "Quarterly revenue report and earnings",
        "Total company sales revenue by quarter",
        "How much revenue was generated in Q1 or Q2",
        "Retrieve financial revenue records from database",
        "Corporate revenue figures for Q4",
        "What was our Q1 2023 revenue",
        "What was our Q2 2023 revenue",
        "What was our Q3 2023 revenue",
        "What was our Q4 2023 revenue",
        "What was our Q1 2024 revenue",
        "What was our Q2 2024 revenue",
        "Check quarterly financial revenue numbers",
        "Q3 2023 Revenue",
    ],
    "faq": [
        "What is your refund policy",
        "How do I reset my account password",
        "What are your business operating hours",
        "Where are your corporate headquarters located",
        "How can I contact customer support help desk",
        "Do you ship orders internationally",
        "What payment methods and credit cards do you accept",
        "How do I track my shipment order",
        "Can I cancel my membership subscription",
        "How is my personal data protected and secured",
        "Can I get a refund for my purchase",
        "I forgot my password how to recover it",
        "When is customer service open",
        "Where is the company headquartered",
    ],
    "general_simple": [
        "What is the capital city of this country",
        "Who wrote this famous book or literature",
        "What is the boiling point temperature of water in Celsius",
        "What is the chemical element symbol in science",
        "How many continents are on Earth",
        "Which planet is known as the red planet",
        "What language is spoken in Brazil",
        "Translate hello into Spanish",
        "Who painted the Mona Lisa",
        "What is the largest ocean on Earth",
        "In which year did World War II end",
        "What is the hardest natural mineral",
        "Which gas do plants absorb during photosynthesis",
        "What is the official currency of Japan",
        "Who is credited with inventing the telephone",
        "What is the capital of France",
        "What is the capital of Japan",
    ],
    "complex_reasoning": [
        "Analyze the macroeconomic impact and systemic consequences of inflation on global supply chains",
        "Compare and contrast the philosophical frameworks of existentialism and stoicism regarding human agency",
        "Perform a multi-criteria architectural evaluation of microservices versus monolithic systems under high concurrency",
        "Explain the theoretical foundations and limitations of Godel's Incompleteness Theorems in mathematical logic",
        "Formulate a comprehensive market entry strategy for an enterprise SaaS startup in healthcare",
        "Evaluate the tradeoffs between Paxos and Raft distributed consensus protocols in distributed storage systems",
        "Synthesize the ethical and governance dilemmas surrounding autonomous weapon systems under international humanitarian law",
        "Provide a structural breakdown of zero-knowledge proofs and their cryptographic scaling potential in Layer 2 blockchains",
        "Critique the methodology and potential selection bias of randomized controlled trials versus observational studies in epidemiology",
        "Develop an incident response and disaster recovery plan for a multi-region ransomware attack",
        "Deconstruct the cognitive biases that lead to groupthink in high-stakes corporate executive decisions",
        "Evaluate the viability of fusion energy commercialization within magnetic confinement tokamak designs",
        "Compare monotonic versus non-monotonic reasoning paradigms in artificial intelligence knowledge representation",
        "Draft a multi-tiered legal and data governance risk assessment for enterprise cross-border data transfers under GDPR",
    ],
}

CATEGORY_TO_PATH: Dict[str, str] = {
    "arithmetic": "calculator",
    "revenue": "sql",
    "faq": "cache",
    "general_simple": "small_llm",
    "complex_reasoning": "large_llm",
    "fallback": "large_llm",
}


class SemanticRouter:
    """
    Initializes the embedding model and anchor vectors.
    Dispatches queries based on cosine similarity and confidence threshold.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", anchors: Optional[Dict[str, List[str]]] = None):
        self.model_name = model_name
        self.anchors = anchors if anchors is not None else DEFAULT_ANCHORS
        self.model = SentenceTransformer(model_name)

        # Pre-compute and normalize embeddings for all anchor strings
        self.anchor_embeddings: Dict[str, np.ndarray] = {}
        for category, phrases in self.anchors.items():
            embeddings = self.model.encode(phrases, convert_to_numpy=True, normalize_embeddings=True)
            self.anchor_embeddings[category] = embeddings

    def get_route(self, query: str, threshold: float) -> str:
        """
        Determines the best route based on cosine similarity.

        Params:
          - query (str): The user's input text
          - threshold (float): Minimum cosine similarity to accept a match

        Returns:
          - (str) The matched category name, or 'fallback' if below threshold
        """
        category, _, _ = self.route_details(query, threshold)
        return category

    def route_details(self, query: str, threshold: float) -> Tuple[str, str, float]:
        """
        Computes cosine similarities, selects max similarity, and checks threshold.

        Returns:
          - (category, path_taken, max_similarity)
        """
        query_vec = self.model.encode(query, convert_to_numpy=True, normalize_embeddings=True)

        best_category = None
        best_sim = -1.0

        for category, anchor_vecs in self.anchor_embeddings.items():
            # Dot product of normalized vectors equals cosine similarity
            sims = np.dot(anchor_vecs, query_vec)
            max_cat_sim = float(np.max(sims))
            if max_cat_sim > best_sim:
                best_sim = max_cat_sim
                best_category = category

        # Threshold Gate: If max_sim >= threshold and category is valid,
        # route to that category. If below threshold, fallback to 'fallback' / 'large_llm'.
        if best_sim >= threshold and best_category is not None:
            chosen_category = best_category
        else:
            chosen_category = "fallback"

        path_taken = CATEGORY_TO_PATH.get(chosen_category, "large_llm")
        return chosen_category, path_taken, best_sim
