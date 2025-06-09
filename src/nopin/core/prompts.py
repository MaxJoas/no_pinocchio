from dataclasses import dataclass
from jinja2 import Template


@dataclass
class Prompts:
    """Store prompt constants using jinja2 templates for dynamic content generation.

    Attributes:
        self_reflection: Prompt Template for self-reflection questions.
        consistency: Template to generate multiple responses for consistency checking.
    """

    self_reflection: Template = Template("""
    1. Question: {{ question }}, Proposed Answer: {{ answer }}
    Is the proposed answer: (A) Correct (B) Incorrect (C) I am not sure.
    The output should strictly use the following template:
    explanation: [insert analysis], answer: [choose one letter from among choices A through C]
    2. Question: {{ question }}, Proposed Answer: {{ answer }}
    Are you really sure the proposed answer is correct?
    Choose again: (A) Correct (B) Incorrect (C) I am not sure.
    The output should strictly use the following template:
    explanation: [insert analysis], answer: [choose one letter from among choices A through C]
    """)
    consistency: Template = Template("""
    Please strictly use the following template to provide answer:
    explanation: [insert step-by-step analysis], answer: [provide
    your answer] + Question: {{ question }}
    """)
