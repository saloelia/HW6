"""Prompting strategy implementations."""

from abc import ABC, abstractmethod
from typing import Optional

from prompt_engineering.data.models import QuestionAnswer, QuestionType


class PromptStrategy(ABC):
    """Abstract base class for prompting strategies.

    This is a building block that defines the interface for all
    prompting strategies. Each strategy transforms a question
    into a complete prompt for the LLM.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the strategy name."""
        pass

    @abstractmethod
    def build_prompt(self, question: QuestionAnswer) -> str:
        """Build the complete prompt for a question.

        Args:
            question: The QA pair to build a prompt for

        Returns:
            The complete prompt string
        """
        pass

    @abstractmethod
    def get_system_prompt(self) -> Optional[str]:
        """Get the system prompt for this strategy.

        Returns:
            System prompt string or None if not using system prompt
        """
        pass


class BaselineStrategy(PromptStrategy):
    """Baseline prompting strategy with minimal instructions.

    Uses atomic prompts - the shortest instruction that accomplishes the task.
    """

    @property
    def name(self) -> str:
        return "baseline"

    def get_system_prompt(self) -> Optional[str]:
        return "You are a helpful assistant. Answer questions concisely."

    def build_prompt(self, question: QuestionAnswer) -> str:
        """Build a minimal baseline prompt."""
        return f"Question: {question.question}\n\nAnswer:"


class FewShotStrategy(PromptStrategy):
    """Few-shot learning strategy with examples.

    Provides 2-3 examples for each question type to guide the model.
    """

    def __init__(self) -> None:
        """Initialize with example sets for each question type."""
        self._examples = self._create_examples()

    @property
    def name(self) -> str:
        return "few_shot"

    def get_system_prompt(self) -> Optional[str]:
        return (
            "You are a helpful assistant. Learn from the examples provided "
            "and answer the new question in the same format."
        )

    def _create_examples(self) -> dict[QuestionType, list[tuple[str, str]]]:
        """Create example sets for each question type."""
        return {
            QuestionType.SENTIMENT: [
                ("The movie was absolutely fantastic!", "positive"),
                ("I hate waiting in long lines.", "negative"),
                ("The weather is cloudy today.", "neutral"),
            ],
            QuestionType.MATH: [
                ("What is 15 + 27?", "42"),
                ("Calculate 8 * 7.", "56"),
                ("What is 100 - 35?", "65"),
            ],
            QuestionType.LOGIC: [
                (
                    "If all cats are mammals, and Whiskers is a cat, is Whiskers a mammal?",
                    "Yes, Whiskers is a mammal.",
                ),
                (
                    "If it's raining, the ground is wet. The ground is wet. Is it raining?",
                    "Not necessarily. The ground could be wet for other reasons.",
                ),
                (
                    "All squares are rectangles. Is every rectangle a square?",
                    "No, not every rectangle is a square.",
                ),
            ],
        }

    def build_prompt(self, question: QuestionAnswer) -> str:
        """Build a few-shot prompt with examples."""
        examples = self._examples.get(question.question_type, [])

        prompt_parts = ["Here are some examples:\n"]

        for i, (q, a) in enumerate(examples, 1):
            prompt_parts.append(f"Example {i}:")
            prompt_parts.append(f"Question: {q}")
            prompt_parts.append(f"Answer: {a}\n")

        prompt_parts.append("Now answer this question:")
        prompt_parts.append(f"Question: {question.question}")
        prompt_parts.append("\nAnswer:")

        return "\n".join(prompt_parts)


class ChainOfThoughtStrategy(PromptStrategy):
    """Chain of Thought (CoT) prompting strategy.

    Encourages step-by-step reasoning before providing the final answer.
    This reduces entropy and improves accuracy on complex tasks.
    """

    @property
    def name(self) -> str:
        return "chain_of_thought"

    def get_system_prompt(self) -> Optional[str]:
        return (
            "You are a helpful assistant that thinks step by step. "
            "Always show your reasoning process before giving the final answer."
        )

    def _get_type_specific_instruction(self, question_type: QuestionType) -> str:
        """Get specific CoT instructions based on question type."""
        instructions = {
            QuestionType.SENTIMENT: (
                "Analyze the tone, word choice, and emotional indicators in the text. "
                "Consider both explicit and implicit sentiment markers."
            ),
            QuestionType.MATH: (
                "Break down the problem into steps. Show your calculations clearly. "
                "Verify your answer by checking your work."
            ),
            QuestionType.LOGIC: (
                "Identify the premises and conclusion. "
                "Evaluate the logical structure and determine validity. "
                "Consider any assumptions or potential fallacies."
            ),
        }
        return instructions.get(question_type, "Think through this step by step.")

    def build_prompt(self, question: QuestionAnswer) -> str:
        """Build a Chain of Thought prompt."""
        instruction = self._get_type_specific_instruction(question.question_type)

        return (
            f"Question: {question.question}\n\n"
            f"Instructions: {instruction}\n\n"
            "Let's think through this step by step:\n"
            "1. First, I will analyze the key elements...\n"
            "2. Then, I will apply relevant reasoning...\n"
            "3. Finally, I will provide my answer.\n\n"
            "Your step-by-step reasoning:"
        )


class ReActStrategy(PromptStrategy):
    """ReAct (Reasoning + Acting) prompting strategy.

    Combines reasoning traces with action steps for complex problem solving.
    Useful for tasks that benefit from external information or tools.
    """

    @property
    def name(self) -> str:
        return "react"

    def get_system_prompt(self) -> Optional[str]:
        return (
            "You are a helpful assistant that uses the ReAct framework. "
            "For each question, alternate between Thought (reasoning), "
            "Action (what to do), and Observation (what you learned). "
            "Continue until you reach a final Answer."
        )

    def build_prompt(self, question: QuestionAnswer) -> str:
        """Build a ReAct prompt with thought-action-observation structure."""
        return (
            f"Question: {question.question}\n\n"
            "Use the following format to solve this problem:\n\n"
            "Thought 1: [Your reasoning about what to do first]\n"
            "Action 1: [The action you would take - analyze, calculate, evaluate, etc.]\n"
            "Observation 1: [What you observe from that action]\n\n"
            "Thought 2: [Your next reasoning step]\n"
            "Action 2: [Your next action]\n"
            "Observation 2: [What you observe]\n\n"
            "(Continue as needed...)\n\n"
            "Final Answer: [Your conclusive answer]\n\n"
            "Begin your response:"
        )
