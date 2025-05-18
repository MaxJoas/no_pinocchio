import re
import json
import requests
import numpy as np
from transformers import pipeline


class BSDetector:
    def __init__(
        self,
        llm_endpoint: str,  # e.g. "http://localhost:11434/api/generate"
        alpha: float = 0.7,  # paper’s α
        beta: float = 0.5,  # paper’s β
        k: int = 5,  # number of samples
        temp: float = 1.0,
    ):  # sampling temperature

        self.llm_name="qwen2.5:0.5b"
        self.llm_endpoint = llm_endpoint
        self.alpha, self.beta, self.k, self.temp = alpha, beta, k, temp
        # NLI cross‑encoder for contradiction scoring
        self.nli = pipeline(
            "text-classification",
            model="cross-encoder/nli-deberta-v3-small",
            device=1,
            top_k=None,
        )

    # def _generate(self, prompt: str, temperature: float = 0.0) -> str:
    #     resp = requests.post(
    #         self.llm,
    #         json={"prompt": prompt, "temperature": temperature, "stream": False},
    #     ).json()
    #     return resp.get("response", "").strip()

    def _generate(self, prompt, temperature=0.0):
        """Generate a response with robust error handling"""
        try:
            # Prepare the request payload
            payload = {
                "model": self.llm_name,
                "prompt": prompt,
                "temperature": temperature,
                "stream": False,
            }

            # Make the API call
            response = requests.post(
                self.llm_endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
            )

            # Check if the request was successful
            if response.status_code != 200:
                raise Exception(
                    f"API returned error status: {response.status_code}, {response.text}"
                )

            # Safely parse the JSON response
            try:
                # Extract the first line to avoid JSON parsing issues
                json_str = response.text.split("\n")[0].strip()
                result = json.loads(json_str)
                return result.get("response", "")
            except json.JSONDecodeError:
                try:
                    # Try using response.json() method
                    result = response.json()
                    return result.get("response", "")
                except Exception:
                    # Manual extraction if all else fails
                    if '"response":"' in response.text:
                        start = response.text.find('"response":"') + 12
                        end = response.text.find('"', start)
                        if end > start:
                            return response.text[start:end]

                    print("Failed to parse Ollama API response")
            return ""

        except Exception as e:
            print(f"Error calling Ollama API: {e}")
            return f"Error: Failed to generate response. Please check if Ollama is running with model {self.llm_name}."

    def _sim(self, ref: str, out: str) -> float:
        # 1 − contradiction probability, averaged in both directions
        print("Calculating similarity score")

        def p_contra(a, b):
            scores = self.nli(f"{a} [SEP] {b}")[0]
            print(f"scores: {scores}")
            return next(d["score"] for d in scores if d["label"] == "contradiction")

        return 1 - (p_contra(ref, out) + p_contra(out, ref)) / 2

    def observed_consistency(self, question: str, reference_answer: str) -> float:
        """
        O = (1/k) Σ [α * s_i + (1−α) * r_i],
        using exactly the supplement’s CoT prompt (Figure 6 a).
        """
        cot_template = (
            "Please strictly use the following template to provide answer: "
            "explanation: [insert step-by-step analysis], answer: [provide your answer] + "
            f"Question: {question}"
        )
        o_vals = []
        for i in range(self.k):
            y_i = self._generate(cot_template, temperature=self.temp)
            print(f"Response {i}: {y_i}")
            s_i = self._sim(reference_answer, y_i)
            r_i = 1.0 if y_i.strip() == reference_answer.strip() else 0.0
            o_vals.append(self.alpha * s_i + (1 - self.alpha) * r_i)
        return float(np.mean(o_vals)) if o_vals else 0.5

    def self_reflection(self, question: str, answer: str) -> float:
        """
        S = average of two multiple‑choice evaluations, with prompts from supplement.
        """
        prompts = [
            # Prompt 1
            (
                f"Question: {question}, Proposed Answer: {answer}. "
                "Is the proposed answer: (A) Correct (B) Incorrect (C) I am not sure. "
                "The output should strictly use the following template: "
                "explanation: [insert analysis], answer: [choose one letter from among choices A through C]"
            ),
            # Prompt 2
            (
                f"Question: {question}, Proposed Answer: {answer}. "
                "Are you really sure the proposed answer is correct? "
                "Choose again: (A) Correct (B) Incorrect (C) I am not sure. "
                "The output should strictly use the following template: "
                "explanation: [insert analysis], answer: [choose one letter from among choices A through C]"
            ),
        ]
        mapping = {"A": 1.0, "B": 0.0, "C": 0.5}
        scores = []
        for i, p in enumerate(prompts):
            out = self._generate(p)
            print(f"Out {i} for self-reflection: {out}")
            m = re.search(r"\b([ABC])\b", out)
            scores.append(mapping.get(m.group(1), 0.5) if m else 0.5)
        return float(np.mean(scores))

    def compute_confidence(self, question: str, answer: str = None):
        # 1) reference answer
        if answer is None:
            answer = self._generate(question, temperature=0.0)
        # 2) observed consistency
        O = self.observed_consistency(question=question, reference_answer=answer)
        # 3) self-reflection certainty
        S = self.self_reflection(question, answer)
        # 4) aggregate
        C = self.beta * O + (1 - self.beta) * S
        return C, answer

    def select_best(self, question: str, candidates: int = 3):
        best_ans, best_conf = None, -1.0
        for _ in range(candidates):
            cand = self._generate(question, temperature=self.temp)
            conf, _ = self.compute_confidence(question, cand)
            if conf > best_conf:
                best_conf, best_ans = conf, cand
        return best_ans, best_conf


if __name__ == "__main__":
    detector = BSDetector(
        llm_endpoint="http://localhost:11434/api/generate",
        alpha=0.7,
        beta=0.5,
        k=5,
        temp=1.0,
    )

    # # Example 1: Provided answer
    # q1 = "What is the capital of France?"
    # a1 = "Paris"
    # conf1, _ = detector.compute_confidence(q1, a1)
    # print(f"Q: {q1}\nA: {a1}\nConfidence: {conf1:.3f}\n")

    # # Example 2: Generate & score
    # q2 = "How many continents are there on Earth?"
    # conf2, gen2 = detector.compute_confidence(q2)
    # print(f"Q: {q2}\nGenerated A: {gen2}\nConfidence: {conf2:.3f}\n")

    # # Example 3: Best‐of‐k selection
    # q3 = "What is the boiling point of water in Celsius?"
    # best_ans, best_conf = detector.select_best(q3, candidates=3)
    # print(f"Q: {q3}\nBest A: {best_ans}\nBest Confidence: {best_conf:.3f}")

    q_hard = "What is 2 + 2?"
    best_ans_hard, best_conf_hard = detector.select_best(q_hard, candidates=3)
    print(f"Q: {q_hard}\nBest A: {best_ans_hard}\nBest Confidence: {best_conf_hard:.3f}")