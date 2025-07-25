from langchain_community.chat_models.openai import ChatOpenAI


class CustomOpenAI(ChatOpenAI):
    def __init__(
        self,
        model_name,
        openai_api_base="http://10.227.91.60:4000/v1",
        openai_api_key="sk-1234",
        max_tokens=2048,
        request_timeout=480,
        **model_kwargs
    ):
        super().__init__(
            openai_api_key=openai_api_key,
            openai_api_base=openai_api_base,
            model_name=model_name,
            max_tokens=max_tokens,
            request_timeout=request_timeout,
            **model_kwargs
        )
        self.model_name = model_name
