import asyncio
import os
import sys
from typing import Any, Dict, List, Optional

from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.base import Chain
from langchain.prompts import PromptTemplate
from langchain.prompts.base import BasePromptTemplate

sys.path.append(os.path.join(os.getcwd(), "kg_construction", "cove"))

import prompts
from custom_llm import CustomOpenAI


class ExecuteVerificationChain(Chain):
    """
    Implements the logic to execute verification questions
    """

    prompt: BasePromptTemplate
    llm: CustomOpenAI
    input_key: str = "verification_questions"
    output_key: str = "verification_answers"
    max_parallel_calls: int = 20

    class Config:
        """Configuration for the pydantic object."""

        extra = "forbid"
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Will be the keys the prompt expects.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return [self.output_key]

    def extract_questions(self, inputs: Dict[str, Any]) -> List[str]:
        # Convert all the verification questions into a list of string
        sub_inputs = {k: v for k, v in inputs.items() if k == self.input_key}
        verification_questions_prompt_value = self.prompt.format_prompt(**sub_inputs)
        verification_questions_str = verification_questions_prompt_value.text
        verification_questions_list = verification_questions_str.split("\n")

        verification_questions_list = [
            q.strip() for q in verification_questions_list if len(q.strip()) != 0
        ]
        return verification_questions_list

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        verification_answers_list = (
            list()
        )  # Will contain the answers of each verification question
        question_answer_pair = (
            ""  # Final output of verification question and answer pair
        )

        verification_questions_list = self.extract_questions(inputs)

        # Setting up prompt for llm self evaluation
        execution_prompt_self_llm = PromptTemplate.from_template(
            prompts.EXECUTE_PLAN_PROMPT_SELF_LLM
        )

        # Executing the verification questions, using self llm
        for question in verification_questions_list:
            execution_prompt_value = execution_prompt_self_llm.format_prompt(
                **{"verification_question": question}
            )
            verification_answer_llm_result = self.llm.generate_prompt(
                [execution_prompt_value],
                callbacks=run_manager.get_child() if run_manager else None,
            )
            verification_answer_str = verification_answer_llm_result.generations[0][
                0
            ].text
            verification_answers_list.append(verification_answer_str)

        # Create verification question and answer pair
        for question, answer in zip(
            verification_questions_list, verification_answers_list
        ):
            question_answer_pair += "Question: {} Answer: {}\n".format(question, answer)

        if run_manager:
            run_manager.on_text("Log something about this run")

        return {self.output_key: question_answer_pair}

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        self.llm.max_tokens = 150
        semaphore = asyncio.Semaphore(self.max_parallel_calls)
        question_answer_pair = (
            ""  # Final output of verification question and answer pair
        )

        verification_questions_list = self.extract_questions(inputs)

        # Setting up prompt for llm self evaluation
        execution_prompt_self_llm = PromptTemplate.from_template(
            prompts.EXECUTE_PLAN_PROMPT_SELF_LLM
        )

        prompt_list = [
            execution_prompt_self_llm.format_prompt(
                **{"verification_question": question}
            )
            for question in verification_questions_list
        ]
        # Whenever you call a language model, or another chain, you should pass
        # a callback manager to it. This allows the inner run to be tracked by
        # any callbacks that are registered on the outer run.
        # You can always obtain a callback manager for this by calling
        # `run_manager.get_child()` as shown below.
        async with semaphore:
            response = await self.llm.agenerate_prompt(
                prompt_list,
                callbacks=run_manager.get_child() if run_manager else None,
                max_tokens=2048,
            )

        verification_answers_list = [r[0].text for r in response.generations]

        for question, answer in zip(
            verification_questions_list, verification_answers_list
        ):
            question_answer_pair += "Question: {} Answer: {}\n".format(question, answer)

        # If you want to log something about this run, you can do so by calling
        # methods on the `run_manager`, as shown below. This will trigger any
        # callbacks that are registered for that event.
        if run_manager:
            await run_manager.on_text("Log something about this run")

        return {self.output_key: question_answer_pair}

    @property
    def _chain_type(self) -> str:
        return "execute_verification_chain"
