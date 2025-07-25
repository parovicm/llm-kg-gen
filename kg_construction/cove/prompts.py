BASELINE_RESPONSE_PROMPT = """Answer the question below which asks for a list of entities (names, places, locations etc). Output should be a numbered list with relevant & concise enitites. NO ADDITIONAL TEXT.

Question: {original_question}

Answer:"""

VERIFICATION_QUESTION_TEMPLATE_PROMPT = """Your task is to create verification questions based on the provided question. NO ADDITIONAL TEXT.
Example Question: Who are some movie actors born in Boston?
Example Verification Question 1: Where was [movie actor] born?
Example Verification Question 2: Was [movie actor] born in [Boston]?
Explanation: In the above example, the verification questions focused only on the ANSWER_ENTITY (name of the movie actor) and QUESTION_ENTITY (birth place).
Similarly, you need to focus on the ANSWER_ENTITY and QUESTION_ENTITY from the actual question and generate verification questions.
Generate at most two different templates.
Actual Question: {original_question}

Verification Question:"""

VERIFICATION_QUESTION_PROMPT = """Your task is to create a series of verification questions based on the given question, the verfication question template and baseline response. NO ADDITIONAL TEXT.
Example Question: Who are some movie actors who were born in Boston?
Example Baseline Response: 1. Matt Damon
2. Chris Evans
Example Verification Question Template 1: Was [movie actor] born in Boston?
Example Verification Question Template 2: Where was [movie actor] born?
Example Verification Questions: 
1. Was Matt Damon born in Boston?
2. Was Chris Evans born in Boston?
1. Where was Matt Damon born?
2. Where was Chris Evans born?

Actual Question: {original_question}
Baseline Response: {baseline_response}
Verification Question Template: {verification_question_template}

Final Verification Questions:"""

EXECUTE_PLAN_PROMPT_SELF_LLM = """Answer the following question correctly.

Question: {verification_question}

Answer:"""

EXECUTE_PLAN_PROMPT = "{verification_questions}"

FINAL_REFINED_ANSWER_PROMPT = """Given the below `Original Query` and `Baseline Answer`, analyze the `Verification Questions & Answers` to obtain the final & refined answer. Output the final answer as a numbered list. Include only correct entities. NO ADDITIONAL TEXT.
Original Query: {original_question}
Baseline Answer: {baseline_response}

Verification Questions & Answer Pairs:
{verification_answers}

Final Refined Answer:"""
