import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import requests

class BSDetector:
    """
    Implementation of BSDetector algorithm from "Quantifying Uncertainty in Answers 
    from any Language Model and Enhancing their Trustworthiness"
    """
    
    def __init__(self, 
                 llm_name="phi",
                 nli_model_name="microsoft/deberta-v3-xsmall-mnli",
                 alpha=0.7,  # Weight for NLI vs exact match
                 beta=0.5,   # Weight for observed consistency vs self-reflection
                 num_samples=5,
                 temperature=1.0):
        """
        Initialize BSDetector with specified models and parameters.
        
        Args:
            llm_name: Ollama model name to use as primary LLM
            nli_model_name: HuggingFace model for NLI (contradiction detection)
            alpha: Weight for NLI similarity vs exact match (paper's α)
            beta: Weight for observed consistency vs self-reflection (paper's β)
            num_samples: Number of samples for observed consistency (paper's k)
            temperature: Temperature for sampling in observed consistency
        """
        self.llm_name = llm_name
        self.api_base = "http://localhost:11434/api"
        
        # Load NLI model from HuggingFace
        self.tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)
        
        # Parameters
        self.alpha = alpha
        self.beta = beta
        self.num_samples = num_samples
        self.temperature = temperature
        
        # Check if CUDA is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nli_model.to(self.device)
        
        # Map labels for NLI model
        # The label mappings might change based on the model
        # DeBERTa MNLI labels: 0 = contradiction, 1 = neutral, 2 = entailment
        self.contradiction_idx = 0
    
    def _generate_response(self, prompt, temperature=0.0):
        """Generate a response from the LLM with specified temperature."""
        response = requests.post(
            f"{self.api_base}/generate",
            json={
                "model": self.llm_name,
                "prompt": prompt,
                "temperature": temperature,
            },
        )
        if response.status_code == 200:
            return response.json()["response"]
        else:
            raise Exception(f"Error generating response: {response.text}")
    
    def _nli_contradiction_score(self, premise: str, hypothesis: str) -> float:
        """
        Use NLI model to calculate contradiction probability between two texts.
        
        Returns:
            1 - contradiction_probability (higher means less contradiction)
        """
        # Prepare inputs for both orders (premise->hypothesis and hypothesis->premise)
        inputs1 = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
        inputs2 = self.tokenizer(hypothesis, premise, return_tensors="pt", truncation=True, max_length=512)
        
        # Move to device
        inputs1 = {k: v.to(self.device) for k, v in inputs1.items()}
        inputs2 = {k: v.to(self.device) for k, v in inputs2.items()}
        
        # Get predictions for both orders
        with torch.no_grad():
            outputs1 = self.nli_model(**inputs1)
            outputs2 = self.nli_model(**inputs2)
        
        # Apply softmax to get probabilities
        probs1 = torch.nn.functional.softmax(outputs1.logits, dim=-1)
        probs2 = torch.nn.functional.softmax(outputs2.logits, dim=-1)
        
        # Get contradiction probabilities
        contradiction_prob1 = probs1[0, self.contradiction_idx].item()
        contradiction_prob2 = probs2[0, self.contradiction_idx].item()
        
        # Average the two directions to mitigate positional bias
        avg_contradiction_prob = (contradiction_prob1 + contradiction_prob2) / 2
        
        # Return 1 - contradiction for similarity score (higher means more similar)
        return 1.0 - avg_contradiction_prob
    
    def _exact_match(self, text1: str, text2: str) -> float:
        """Calculate exact match (indicator function r_i in paper)."""
        return 1.0 if text1.strip() == text2.strip() else 0.0
    
    def _compute_similarity(self, reference: str, generated: str) -> float:
        """
        Compute similarity combining NLI and exact match.
        
        Formula from paper: o_i = α*s_i + (1-α)*r_i
        Where:
        - s_i is the NLI similarity (1 - contradiction probability)
        - r_i is the exact match indicator (1 if match, 0 otherwise)
        - α is the weight parameter
        """
        # Get NLI similarity score (s_i in paper)
        nli_sim = self._nli_contradiction_score(reference, generated)
        
        # Get exact match indicator (r_i in paper)
        exact_match = self._exact_match(reference, generated)
        
        # Combine with alpha weight (o_i in paper)
        # o_i = α*s_i + (1-α)*r_i
        similarity = self.alpha * nli_sim + (1 - self.alpha) * exact_match
        
        return similarity
    
    def observed_consistency(self, question: str, reference_answer: str) -> float:
        """
        Measure observed consistency by generating multiple answers.
        
        Paper formula: O = (1/k) * Σo_i
        Where:
        - k is the number of samples
        - o_i is the similarity score for each sampled answer
        """
        # Template for Chain-of-Thought prompting
        cot_template = f"""Please strictly use the following template to provide answer:
explanation: [insert step-by-step analysis], answer: [provide your answer]

Question: {question}"""
        
        similarities = []
        for _ in range(self.num_samples):
            # Generate response with temperature sampling
            alternative_answer = self._generate_response(cot_template, temperature=self.temperature)
            
            # Extract the final answer (simplified, assuming format is followed)
            if "answer:" in alternative_answer.lower():
                answer_part = alternative_answer.lower().split("answer:")[1].strip()
            else:
                answer_part = alternative_answer
            
            # Compute similarity (o_i in paper)
            similarity = self._compute_similarity(reference_answer, answer_part)
            similarities.append(similarity)
        
        # Return average similarity as observed consistency (O in paper)
        # O = (1/k) * Σo_i
        return np.mean(similarities)
    
    def self_reflection_certainty(self, question: str, reference_answer: str) -> float:
        """
        Measure self-reflection certainty by asking the model to evaluate its answer.
        
        Paper formula: S = (score_1 + score_2 + ... + score_n) / n
        Where:
        - n is the number of reflection questions
        - scores are: A=1.0, B=0.0, C=0.5
        """
        # First reflection prompt
        reflection_prompt1 = f"""Question: {question}, Proposed Answer: {reference_answer}
Is the proposed answer: (A) Correct (B) Incorrect (C) I am not sure.
The output should strictly use the following template:
explanation: [insert analysis], answer: [choose one letter from among choices A through C]"""
        
        # Second reflection prompt
        reflection_prompt2 = f"""Question: {question}, Proposed Answer: {reference_answer}
Are you really sure the proposed answer is correct?
Choose again: (A) Correct (B) Incorrect (C) I am not sure.
The output should strictly use the following template:
explanation: [insert analysis], answer: [choose one letter from among choices A through C]"""
        
        # Get responses
        reflection1 = self._generate_response(reflection_prompt1)
        reflection2 = self._generate_response(reflection_prompt2)
        
        # Extract answers (simplified parsing)
        def extract_letter(reflection):
            if "answer:" in reflection.lower():
                letter = reflection.lower().split("answer:")[1].strip()[0]
                if letter in "abc":
                    return letter
            # Fallback to searching for the letter
            for letter in "abc":
                if f"({letter.upper()})" in reflection:
                    return letter
            return "c"  # Default to "not sure" if parsing fails
        
        letter1 = extract_letter(reflection1)
        letter2 = extract_letter(reflection2)
        
        # Convert to numerical values: A=1.0, B=0.0, C=0.5
        values = {"a": 1.0, "b": 0.0, "c": 0.5}
        score1 = values.get(letter1, 0.5)
        score2 = values.get(letter2, 0.5)
        
        # Return average score (S in paper)
        # S = (score_1 + score_2) / 2
        return (score1 + score2) / 2
    
    def compute_confidence(self, question: str, answer: Optional[str] = None) -> Tuple[float, str]:
        """
        Compute overall confidence score for an answer.
        
        Paper formula: C = β*O + (1-β)*S
        Where:
        - O is the observed consistency
        - S is the self-reflection certainty
        - β is the weighting parameter
        
        Returns:
            Tuple of (confidence_score, answer)
        """
        # If no answer provided, generate one with temperature=0
        if answer is None:
            answer = self._generate_response(question)
        
        # Compute the two components
        consistency = self.observed_consistency(question, answer)
        reflection = self.self_reflection_certainty(question, answer)
        
        # Combine with beta weighting (C in paper)
        # C = β*O + (1-β)*S
        confidence = self.beta * consistency + (1 - self.beta) * reflection
        
        return confidence, answer
    
    def select_best_answer(self, question: str, num_candidates: int = 3) -> Tuple[str, float]:
        """
        Generate multiple candidate answers and select the one with highest confidence.
        
        This implements the method described in "Application: Generating More Reliable
        Answers from any LLM" section of the paper.
        
        Args:
            question: The question to answer
            num_candidates: Number of candidate answers to generate
            
        Returns:
            Tuple of (best_answer, confidence_score)
        """
        candidates = []
        
        # Generate multiple candidate answers with temperature sampling
        for _ in range(num_candidates):
            candidate = self._generate_response(question, temperature=self.temperature)
            confidence, _ = self.compute_confidence(question, candidate)
            candidates.append((candidate, confidence))
        
        # Select the answer with highest confidence
        best_candidate = max(candidates, key=lambda x: x[1])
        
        return best_candidate


# Example usage
if __name__ == "__main__":
    # Make sure you have:
    # 1. Installed transformers: pip install transformers torch
    # 2. Started Ollama with a model like "phi": ollama run phi
    
    detector = BSDetector(
        llm_name="phi",                          # Local Ollama model
        nli_model_name="microsoft/deberta-v3-xsmall-mnli",  # Small NLI model
        alpha=0.7,                               # Weight for NLI vs exact match
        beta=0.5,                                # Weight for observed consistency vs self-reflection
        num_samples=5,                           # Number of samples for consistency
        temperature=1.0                          # Temperature for sampling
    )
    
    # Example 1: Correct answer
    question1 = "What is the capital of France?"
    answer1 = "Paris"
    
    confidence1, _ = detector.compute_confidence(question1, answer1)
    print(f"Question: {question1}")
    print(f"Answer: {answer1}")
    print(f"Confidence: {confidence1:.3f}")
    
    # Example 2: Wrong answer
    question2 = "What is the capital of France?"
    answer2 = "London"
    
    confidence2, _ = detector.compute_confidence(question2, answer2)
    print(f"\nQuestion: {question2}")
    print(f"Answer: {answer2}")
    print(f"Confidence: {confidence2:.3f}")
    
    # Example 3: Select best answer from multiple candidates
    question3 = "What is the distance from Earth to the Moon in kilometers?"
    
    best_answer, confidence3 = detector.select_best_answer(question3, num_candidates=3)
    print(f"\nQuestion: {question3}")
    print(f"Best answer: {best_answer}")
    print(f"Confidence: {confidence3:.3f}")
