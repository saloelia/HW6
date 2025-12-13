"""Tests for prompting strategies."""

import pytest

from prompt_engineering.data.models import QuestionAnswer, QuestionType
from prompt_engineering.prompts.strategies import (
    BaselineStrategy,
    ChainOfThoughtStrategy,
    FewShotStrategy,
    PromptStrategy,
    ReActStrategy,
)


@pytest.fixture
def sentiment_question() -> QuestionAnswer:
    """Create a sentiment question for testing."""
    return QuestionAnswer(
        id="sent_test",
        question_type=QuestionType.SENTIMENT,
        question="The movie was great!",
        expected_answer="positive",
    )


@pytest.fixture
def math_question() -> QuestionAnswer:
    """Create a math question for testing."""
    return QuestionAnswer(
        id="math_test",
        question_type=QuestionType.MATH,
        question="What is 5 + 7?",
        expected_answer="12",
    )


@pytest.fixture
def logic_question() -> QuestionAnswer:
    """Create a logic question for testing."""
    return QuestionAnswer(
        id="logic_test",
        question_type=QuestionType.LOGIC,
        question="All dogs are animals. Rex is a dog. Is Rex an animal?",
        expected_answer="yes",
    )


class TestBaselineStrategy:
    """Tests for BaselineStrategy."""

    def test_name(self):
        """Test strategy name."""
        strategy = BaselineStrategy()
        assert strategy.name == "baseline"

    def test_system_prompt(self):
        """Test system prompt is set."""
        strategy = BaselineStrategy()
        system_prompt = strategy.get_system_prompt()
        assert system_prompt is not None
        assert "helpful assistant" in system_prompt.lower()

    def test_build_prompt_sentiment(self, sentiment_question):
        """Test prompt building for sentiment question."""
        strategy = BaselineStrategy()
        prompt = strategy.build_prompt(sentiment_question)

        assert "Question:" in prompt
        assert sentiment_question.question in prompt
        assert "Answer:" in prompt

    def test_build_prompt_math(self, math_question):
        """Test prompt building for math question."""
        strategy = BaselineStrategy()
        prompt = strategy.build_prompt(math_question)

        assert math_question.question in prompt

    def test_prompt_is_minimal(self, sentiment_question):
        """Test that baseline prompt is minimal (atomic)."""
        strategy = BaselineStrategy()
        prompt = strategy.build_prompt(sentiment_question)

        lines = [l for l in prompt.split("\n") if l.strip()]
        assert len(lines) <= 3

    def test_is_prompt_strategy(self):
        """Test that BaselineStrategy is a PromptStrategy."""
        strategy = BaselineStrategy()
        assert isinstance(strategy, PromptStrategy)


class TestFewShotStrategy:
    """Tests for FewShotStrategy."""

    def test_name(self):
        """Test strategy name."""
        strategy = FewShotStrategy()
        assert strategy.name == "few_shot"

    def test_system_prompt(self):
        """Test system prompt mentions examples."""
        strategy = FewShotStrategy()
        system_prompt = strategy.get_system_prompt()
        assert "example" in system_prompt.lower()

    def test_build_prompt_includes_examples(self, sentiment_question):
        """Test prompt includes examples."""
        strategy = FewShotStrategy()
        prompt = strategy.build_prompt(sentiment_question)

        assert "Example" in prompt
        assert "Example 1" in prompt

    def test_sentiment_examples_included(self, sentiment_question):
        """Test sentiment examples are included for sentiment questions."""
        strategy = FewShotStrategy()
        prompt = strategy.build_prompt(sentiment_question)

        assert "positive" in prompt.lower() or "negative" in prompt.lower()

    def test_math_examples_included(self, math_question):
        """Test math examples are included for math questions."""
        strategy = FewShotStrategy()
        prompt = strategy.build_prompt(math_question)

        assert any(str(num) in prompt for num in [42, 56, 65])

    def test_logic_examples_included(self, logic_question):
        """Test logic examples are included for logic questions."""
        strategy = FewShotStrategy()
        prompt = strategy.build_prompt(logic_question)

        assert "mammal" in prompt.lower() or "rectangle" in prompt.lower()

    def test_target_question_at_end(self, sentiment_question):
        """Test that target question appears at end."""
        strategy = FewShotStrategy()
        prompt = strategy.build_prompt(sentiment_question)

        assert prompt.endswith("Answer:")
        assert sentiment_question.question in prompt


class TestChainOfThoughtStrategy:
    """Tests for ChainOfThoughtStrategy."""

    def test_name(self):
        """Test strategy name."""
        strategy = ChainOfThoughtStrategy()
        assert strategy.name == "chain_of_thought"

    def test_system_prompt_mentions_reasoning(self):
        """Test system prompt mentions step by step."""
        strategy = ChainOfThoughtStrategy()
        system_prompt = strategy.get_system_prompt()
        assert "step by step" in system_prompt.lower()

    def test_build_prompt_has_steps(self, math_question):
        """Test prompt includes step markers."""
        strategy = ChainOfThoughtStrategy()
        prompt = strategy.build_prompt(math_question)

        assert "step by step" in prompt.lower()
        assert "1." in prompt or "First" in prompt

    def test_sentiment_specific_instruction(self, sentiment_question):
        """Test sentiment-specific instructions."""
        strategy = ChainOfThoughtStrategy()
        prompt = strategy.build_prompt(sentiment_question)

        assert "tone" in prompt.lower() or "sentiment" in prompt.lower()

    def test_math_specific_instruction(self, math_question):
        """Test math-specific instructions."""
        strategy = ChainOfThoughtStrategy()
        prompt = strategy.build_prompt(math_question)

        assert "calculation" in prompt.lower() or "step" in prompt.lower()

    def test_logic_specific_instruction(self, logic_question):
        """Test logic-specific instructions."""
        strategy = ChainOfThoughtStrategy()
        prompt = strategy.build_prompt(logic_question)

        assert "premise" in prompt.lower() or "logical" in prompt.lower()


class TestReActStrategy:
    """Tests for ReActStrategy."""

    def test_name(self):
        """Test strategy name."""
        strategy = ReActStrategy()
        assert strategy.name == "react"

    def test_system_prompt_mentions_react(self):
        """Test system prompt mentions ReAct components."""
        strategy = ReActStrategy()
        system_prompt = strategy.get_system_prompt()

        assert "thought" in system_prompt.lower()
        assert "action" in system_prompt.lower()
        assert "observation" in system_prompt.lower()

    def test_build_prompt_has_format(self, math_question):
        """Test prompt includes Thought-Action-Observation format."""
        strategy = ReActStrategy()
        prompt = strategy.build_prompt(math_question)

        assert "Thought" in prompt
        assert "Action" in prompt
        assert "Observation" in prompt

    def test_has_final_answer_marker(self, sentiment_question):
        """Test prompt includes Final Answer marker."""
        strategy = ReActStrategy()
        prompt = strategy.build_prompt(sentiment_question)

        assert "Final Answer" in prompt

    def test_question_included(self, logic_question):
        """Test that question is included in prompt."""
        strategy = ReActStrategy()
        prompt = strategy.build_prompt(logic_question)

        assert logic_question.question in prompt


class TestStrategyAbstraction:
    """Tests for the abstract PromptStrategy interface."""

    def test_all_strategies_have_name(self):
        """Test all strategies have unique names."""
        strategies = [
            BaselineStrategy(),
            FewShotStrategy(),
            ChainOfThoughtStrategy(),
            ReActStrategy(),
        ]
        names = [s.name for s in strategies]

        assert len(names) == len(set(names))

    def test_all_strategies_build_prompts(self, sentiment_question):
        """Test all strategies can build prompts."""
        strategies = [
            BaselineStrategy(),
            FewShotStrategy(),
            ChainOfThoughtStrategy(),
            ReActStrategy(),
        ]

        for strategy in strategies:
            prompt = strategy.build_prompt(sentiment_question)
            assert isinstance(prompt, str)
            assert len(prompt) > 0

    def test_all_strategies_have_system_prompts(self):
        """Test all strategies provide system prompts."""
        strategies = [
            BaselineStrategy(),
            FewShotStrategy(),
            ChainOfThoughtStrategy(),
            ReActStrategy(),
        ]

        for strategy in strategies:
            system_prompt = strategy.get_system_prompt()
            assert system_prompt is None or isinstance(system_prompt, str)
