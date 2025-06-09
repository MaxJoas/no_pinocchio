import re
from datetime import datetime
from typing import List
import numpy as np

from nopin.clients.llm import LLMClient
from nopin.clients.nli import NLIClient
from nopin.config.settings import Config
from nopin.core.prompts import Prompts


class NoPinocchio:
    """Main class for confidence estimation using self-reflection and consistency.

    Attributes:
        quesion: user request to the LLM
        answer: initial output of the LLM to the user quesion

    """

    def __init__(
        self,
        *,
        prompts: Prompts,
        llm_client: LLMClient,
        nli_client: NLIClient,
        k: int = 5,
        alpha: float = 0.5,
        beta: float = 0.5,
    ):
        """Initialize NoPinocchio instance.

        Args:
            prompts: Prompt templates.
            llm_client: LLM client instance.
            nli_client: NLI client instance.
            question: Question to analyze.
            k: Number of responses to sample for consistency.
            alpha: Weight for semantic consistency vs exact match.
            beta: Weight for observed consistency vs self-reflection.
        """
        self._prompts = prompts
        self._llm_client = llm_client
        self._nli_client = nli_client
        self._k = k
        self.question: str
        self._alpha = alpha
        self._beta = beta
        self.answer: str = ""

    @classmethod
    def from_config(cls, *, config: Config) -> "NoPinocchio":
        """Create NoPinocchio instance from configuration.

        Args:
            config: Configuration object.
            question: Question to analyze.

        Returns:
            NoPinocchio instance.
        """
        prompts = Prompts()  # Use default prompts with your original templates
        llm_client = LLMClient(config=config)
        nli_client = NLIClient(config=config)

        return cls(
            prompts=prompts,
            llm_client=llm_client,
            nli_client=nli_client,
            k=config.nopinocchio.k,
            alpha=config.nopinocchio.alpha,
            beta=config.nopinocchio.beta,
        )

    def _get_self_reflection_prompt(self) -> str:
        """Generate self-reflection prompt.

        Returns:
            Rendered self-reflection prompt.
        """
        return self._prompts.self_reflection.render(
            question=self.question, answer=self.answer
        )

    def _get_consistency_prompt(self, *, sampled_answer: str) -> str:
        """Generate consistency prompt.

        Args:
            sampled_answer: Sampled answer for consistency checking.

        Returns:
            Rendered consistency prompt.
        """
        return self._prompts.consistency.render(
            question=self.question, answer=sampled_answer
        )

    def _get_k_responses(self) -> List[str]:
        """Generate k responses for consistency checking.

        Returns:
            List of sampled responses.
        """
        sampled_answers = []
        for _ in range(self._k):
            response = self._llm_client.chat(question=self.question, temperature="max")
            sampled_answers.append(response)
        return sampled_answers

    def _calc_self_reflection_score(self) -> float:
        """Calculate self-reflection score.

        Returns:
            Self-reflection score between 0 and 1.
        """
        prompt = self._get_self_reflection_prompt()
        sr_answer = self._llm_client.chat(question=prompt, temperature="min")

        rendered_answers = re.findall(
            r"answer:\s*([ABC])", sr_answer, flags=re.IGNORECASE
        )

        def map_to_score(letter: str) -> float:
            """Map letter to numerical score."""
            return {"A": 1.0, "B": 0.0, "C": 0.5}.get(letter.upper(), 0.5)

        scores = [map_to_score(letter=answer) for answer in rendered_answers]

        if len(scores) == 0:
            # Default to uncertain if no valid answers found
            return 0.5

        return float(np.mean(scores))

    def _calc_pairwise_consistency(self, *, sampled_answer: str) -> float:
        """Calculate pairwise consistency between main answer and sampled answer.

        Args:
            sampled_answer: Sampled answer to compare with.

        Returns:
            Pairwise consistency score.
        """
        y_yi = f"{self.answer} [SEP] {sampled_answer}"
        yi_y = f"{sampled_answer} [SEP] {self.answer}"

        res_y_yi = self._nli_client(y_yi)
        res_yi_y = self._nli_client(yi_y)

        # Extract contradiction probabilities
        p_contra = next(
            (item["score"] for item in res_y_yi[0] if item["label"] == "contradiction"),
            0.0,
        )
        p_contra_prime = next(
            (item["score"] for item in res_yi_y[0] if item["label"] == "contradiction"),
            0.0,
        )

        # Semantic consistency score
        si = 0.5 * ((1 - p_contra) + (1 - p_contra_prime))

        # Exact match indicator
        ri = 1.0 if self.answer == sampled_answer else 0.0

        return self._alpha * si + (1 - self._alpha) * ri

    def _calc_observed_consistency(self, *, k_sampled_answers: List[str]) -> float:
        """Calculate observed consistency across k sampled answers.

        Args:
            k_sampled_answers: List of sampled answers.

        Returns:
            Average observed consistency score.
        """
        consistency_scores = []
        for answer in k_sampled_answers:
            score = self._calc_pairwise_consistency(sampled_answer=answer)
            consistency_scores.append(score)

        return float(np.mean(consistency_scores))

    def _calc_confidence_score(
        self, *, observed_consistency: float, self_reflection: float
    ) -> float:
        """Calculate final confidence score.

        Args:
            observed_consistency: Observed consistency score.
            self_reflection: Self-reflection score.

        Returns:
            Final confidence score.
        """
        return self._beta * observed_consistency + (1 - self._beta) * self_reflection

    def get_confidence(self) -> float:
        """Get confidence score for the question.

        Returns:
            Confidence score between 0 and 1.
        """
        # Get the main answer
        self.answer = self._llm_client.chat(question=self.question, temperature="min")

        # Sample k additional responses
        k_sampled_answers = self._get_k_responses()

        # Calculate consistency and self-reflection scores
        observed_consistency = self._calc_observed_consistency(
            k_sampled_answers=k_sampled_answers
        )
        self_reflection = self._calc_self_reflection_score()

        # Return final confidence score
        return self._calc_confidence_score(
            observed_consistency=observed_consistency, self_reflection=self_reflection
        )

    def analyze_question(self, *, question: str) -> dict:
        self.question = question
        self.answer = ""
        confidence_score = self.get_confidence()
        return {
            "question": question,
            "answer": self.answer,
            "confidence_score": confidence_score,
            "timestamp": datetime.now().isoformat(),
        }
