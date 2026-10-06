import json

import httpx

from consult_to_note.config import Settings
from consult_to_note.llm import NIMClient, complete_structured
from consult_to_note.schemas import JudgeVerdict


async def test_nim_request_uses_guided_json_and_thinking_switch():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        seen["auth"] = request.headers["authorization"]
        return httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": seen["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": '{"supported": true, "reason": "ok"}'},
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17},
            },
        )

    settings = Settings(
        api_key="nvapi-test",
        base_url="https://integrate.api.nvidia.com/v1",
        model_fast="nvidia/fast",
        model_reasoning="nvidia/big",
    )
    client = NIMClient(settings, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    verdict = await complete_structured(client, [{"role": "user", "content": "hi"}], JudgeVerdict, step="t")

    assert verdict.supported is True
    assert seen["model"] == "nvidia/fast"
    assert seen["guided_json"]["title"] == "JudgeVerdict"
    assert seen["chat_template_kwargs"] == {"enable_thinking": False}
    assert seen["auth"] == "Bearer nvapi-test"
    assert client.calls[0].prompt_tokens == 12
