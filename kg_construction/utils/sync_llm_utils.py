import time

from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


class LLM:
    def __init__(
        self, temperature=0.1, base_url="http://10.227.91.60:4000/v1", api_key="sk-1234"
    ):
        self.temperature = temperature
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    def print_models(self):
        res = self.client.models.list()
        for m in sorted(res.data, key=lambda x: x.id):
            print(m)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
    def call(
        self, user_prompt, system_prompt="", model_name="llama3.1-70b", temperature=None
    ):
        start = time.time()

        if temperature:
            temp = temperature
        else:
            temp = self.temperature

        res = self.client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temp,
            stream=False,
        )

        res = res.choices[0].message.content

        end = time.time()
        response_time = str(end - start)
        return res, response_time

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
    def call_without_stream(
        self,
        user_prompt,
        system_prompt="user",
        model_name="llama3.1-70b",
        temperature=None,
        seed=None,
    ):
        start = time.time()

        if temperature:
            temp = temperature
        else:
            temp = self.temperature

        if seed:
            s = seed
        else:
            s = 12345

        res = self.client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temp,
            seed=s,
        )

        response_content = res.choices[0].message.content
        end = time.time()
        response_time = str(end - start)
        return response_content, response_time

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
    def call_with_formater(
        self,
        format_model,
        user_prompt,
        system_prompt="user",
        model_name="llama3.1-70b",
        temperature=None,
        seed=None,
    ):
        start = time.time()

        if temperature:
            temp = temperature
        else:
            temp = self.temperature

        res = self.client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temp,
            stream=False,
            extra_body={
                "guided_json": format_model,
                "guided_decoding_backend": "outlines",
            },
            seed=seed,
        )

        res = res.choices[0].message.content

        end = time.time()
        response_time = str(end - start)
        return res, response_time

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
    def call_with_choice(
        self,
        choices,
        user_prompt,
        system_prompt="user",
        model_name="llama3.1-70b",
        temperature=None,
        seed=None,
    ):
        start = time.time()

        if temperature:
            temp = temperature
        else:
            temp = self.temperature

        res = self.client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temp,
            stream=False,
            extra_body={"guided_choice": choices},
            seed=seed,
        )
        res = res.choices[0].message.content

        end = time.time()
        response_time = str(end - start)
        return res, response_time

    def temp(self):
        return str(self.temperature)


if __name__ == "__main__":
    model = LLM(0.2)
    model.print_models()
