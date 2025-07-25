import os
import sys

from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

sys.path.append(os.path.join(os.getcwd(), "kg_construction", "cove"))

import prompts
from .execute_verification_chain import ExecuteVerificationChain


class EntityListCOVEChain:
    def __init__(self, llm, baseline_response_exists=False, max_parallel_calls=20):
        self.llm = llm
        self.baseline_response_exists = baseline_response_exists
        self.max_parallel_calls = max_parallel_calls

    def __call__(self):
        # Create baseline response chain
        baseline_response_prompt_template = PromptTemplate.from_template(
            prompts.BASELINE_RESPONSE_PROMPT
        )
        baseline_response_chain = (
            baseline_response_prompt_template | self.llm | StrOutputParser()
        )

        # Create verification chain
        verification_question_template_prompt_template = PromptTemplate.from_template(
            prompts.VERIFICATION_QUESTION_TEMPLATE_PROMPT
        )
        verification_question_template_chain = (
            verification_question_template_prompt_template
            | self.llm
            | StrOutputParser()
        )

        # Create verification questions
        verification_question_generation_prompt_template = PromptTemplate.from_template(
            prompts.VERIFICATION_QUESTION_PROMPT
        )
        verification_question_generation_chain = (
            verification_question_generation_prompt_template
            | self.llm
            | StrOutputParser()
        )

        # Create execution verification
        execute_verification_question_prompt_template = PromptTemplate.from_template(
            prompts.EXECUTE_PLAN_PROMPT
        )
        execute_verification_question_chain = ExecuteVerificationChain(
            llm=self.llm,
            prompt=execute_verification_question_prompt_template,
            output_key="verification_answers",
            max_parallel_calls=self.max_parallel_calls,
        )

        # Create final refined response
        final_answer_prompt_template = PromptTemplate.from_template(
            prompts.FINAL_REFINED_ANSWER_PROMPT
        )
        final_answer_chain = final_answer_prompt_template | self.llm | StrOutputParser()

        cove_chain = (
            RunnablePassthrough.assign(
                verification_question_template=verification_question_template_chain
            )
            | RunnablePassthrough.assign(
                verification_questions=verification_question_generation_chain
            )
            | RunnablePassthrough.assign(
                verification_answers=execute_verification_question_chain
            )
            | RunnablePassthrough.assign(final_answer=final_answer_chain)
        )

        if not self.baseline_response_exists:
            cove_chain = (
                RunnablePassthrough.assign(baseline_response=baseline_response_chain)
                | cove_chain
            )

        return cove_chain
