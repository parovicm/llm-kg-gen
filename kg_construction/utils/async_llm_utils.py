import asyncio

import openai
import urllib3
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm.asyncio import tqdm

urllib3.disable_warnings(urllib3.connectionpool.InsecureRequestWarning)


# OpenAI ChatGPT interface
def get_user_message(txt):
    return {"role": "user", "content": txt}


def get_system_message(txt):
    return {"role": "system", "content": txt}


def get_assistant_message(txt):
    return {"role": "assistant", "content": txt}


class Openai_Compatible:
    def __init__(
        self,
        base_url="http://10.227.91.60:4000/v1",
        api_key="sk-1234",
        temperature=0.2,
        model="llama3.1-70b",
        retain_history=False,
    ) -> None:

        self.use_async = False

        if api_key is None:
            api_key = "EMPTY"

        self.retain_history = retain_history
        # Get the local model from the client
        self.client = openai.OpenAI(
            # defaults to os.environ.get("OPENAI_API_KEY")
            api_key=api_key,
            base_url=base_url,
        )
        self.model = model
        print(f"Currently using the LLM: {self.model}")

        self.temperature = temperature
        self._conversation = []

    def clear_history(self):
        self._conversation = []

    @property
    def models(self):
        return [x.id for x in self.client.models.list()]

    def __repr__(self):
        return "\n".join(self.models)

    def __call__(self, user_prompt, system_prompt=""):
        if not self.retain_history:
            self.clear_history()

        message = get_user_message(user_prompt)
        system_info = get_system_message(system_prompt)
        self._conversation.append(system_info)
        self._conversation.append(message)

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=self._conversation,
            temperature=self.temperature,
        )
        response = completion.choices[0].message.content
        self._conversation.append(get_assistant_message(response))
        return response


class Async_Openai_Compatible:
    def __init__(
        self,
        base_url="http://10.227.91.60:4000/v1",
        temperature=0.2,
        api_key="sk-1234",
        model="llama3.1-70b",
        retain_history=False,
    ) -> None:
        self.use_async = True

        if api_key is None:
            api_key = "EMPTY"

        self.retain_history = retain_history
        # Get the local model from the client
        self.client = openai.AsyncOpenAI(
            # defaults to os.environ.get("OPENAI_API_KEY")
            api_key=api_key,
            base_url=base_url,
        )

        while True:
            try:
                models = self.client.models.list()
                break
            except:
                continue

        if model is None:
            raise ValueError("Model Cannot be None")
        else:
            self.model = model
        print(f"Currently using the LLM: {self.model}")

        self.temperature = temperature
        self._conversation = []

    def clear_history(self):
        self._conversation = []

    async def __call__(
        self,
        user_prompt,
        system_prompt="",
        format_enforcing=None,
        choice_enforcing=None,
    ):
        if not self.retain_history:
            self.clear_history()

        message = get_user_message(user_prompt)
        system_info = get_system_message(system_prompt)
        self._conversation.append(system_info)
        self._conversation.append(message)

        if format_enforcing:
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=self._conversation,
                temperature=self.temperature,
                extra_body={
                    "guided_json": format_enforcing,
                    "guided_decoding_backend": "outlines",
                },
            )
        else:
            if choice_enforcing:
                completion = await self.client.chat.completions.create(
                    model=self.model,
                    messages=self._conversation,
                    temperature=self.temperature,
                    extra_body={"guided_choice": choice_enforcing},
                )
            else:
                completion = await self.client.chat.completions.create(
                    model=self.model,
                    messages=self._conversation,
                    temperature=self.temperature,
                    max_tokens=2048,
                )

        response = completion.choices[0].message.content
        self._conversation.append(get_assistant_message(response))
        return response


@retry(
    wait=wait_random_exponential(min=0.5, max=60),
    stop=stop_after_attempt(10),
    before_sleep=print,
    retry_error_callback=lambda _: None,
)
async def get_completion(
    user_prompt,
    llm,
    semaphore,
    system_prompt="",
    format_enforcing=None,
    choice_enforcing=None,
):
    async with semaphore:
        out = await llm(
            user_prompt,
            system_prompt=system_prompt,
            format_enforcing=format_enforcing,
            choice_enforcing=choice_enforcing,
        )
        return out


async def get_completion_list(
    content_list,
    llm: Async_Openai_Compatible,
    max_parallel_calls,
    choice_enforcing=None,
    format_enforcing=None,
):
    """Order preserved async execution of prompts."""

    semaphore = asyncio.Semaphore(value=max_parallel_calls)

    return await tqdm.gather(
        *[
            get_completion(
                content["user_message"],
                llm,
                semaphore,
                content["system_message"],
                choice_enforcing=choice_enforcing,
                format_enforcing=format_enforcing,
            )
            for content in content_list
        ]
    )


async def main():
    content_list = [
        {"system_message": "", "user_message": "List actors born in Boston."},
        {"system_message": "", "user_message": "List books written by George Orwell."},
    ] * 5
    llm = Async_Openai_Compatible(model="llama3.1-70b")
    answers = await get_completion_list(content_list, llm, 10)
    meta_data = [{"a": 1}] * 10
    print(list(zip(answers, meta_data)))
