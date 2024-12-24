import json
import re
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Optional

from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

from approaches.approach import Approach


class ChatApproach(Approach, ABC):
    query_prompt_few_shots: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": "Tell me more about the Ignite 2024"},
        {"role": "assistant", "content": "Summarize the event topics"},
        {"role": "user", "content": "What is the new in AI?"},
        {"role": "assistant", "content": "Look for content related to AI"},
    ]
    NO_RESPONSE = "0"

   
    query_prompt_template = """You are a content retrieval assistant tasked with finding information specifically related to the Microsoft Ignite 2024 event using the provided index data. Do not answer or respond to queries that are outside the defined scope of Microsoft Ignite 2024.

# Instructions

- Use the attached index data only to locate and extract relevant content about the Microsoft Ignite 2024 event. 
- If the query is out of scope or unrelated to Microsoft Ignite 2024, respond with a polite message indicating that the query is not relevant to the task or dataset.
- Avoid generating speculative responses or using information outside of the provided index data.
- As all announcement include links, Always include the links to the response to further readings

# Steps

1. **Understand the Query**  
   - Ensure the query is specifically about Microsoft Ignite 2024.  
   - If unclear, prioritize locating content using keywords such as "Microsoft Ignite," "2024," "sessions," "keynotes," "schedule," or other event-specific terms.  
   - If the query doesn't align with the event or uses generic terms without context, flag it as out of scope.  

2. **Search the Index Data**  
   - Look for matches or references in the index data that correspond to the user's query.  
   - Extract the most accurate and relevant information based on the available context.

3. **Validate Results**  
   - Ensure the information is directly tied to Microsoft Ignite 2024 and avoid generic Microsoft or unrelated event data.  
   - If there is partial information, clarify accordingly but stay within scope.

4. **Respond to User**  
   - If relevant data is found: Provide a concise and direct response in relation to the query.  
   - If no relevant data is found in the index: Reply politely that the requested information isn't available in the provided scope.  

# Output Format

- **Relevant Query Match**: A concise response with relevant data points found in the index, formatted in plain text or a bulleted list for clarity.
- **Out of Scope Message**: "I can only assist with queries specifically related to Microsoft Ignite 2024 using the provided index. Could you adjust your query to fall within this scope?"

# Example

### Input Query 1:
"What are the keynotes at Microsoft Ignite 2024?"

### Output:
- "Based on the index, the keynotes at Microsoft Ignite 2024 include:  
  - [Speaker Name/Topic 1]  
  - [Speaker Name/Topic 2]  
  Please let me know if you’d like further details on these sessions."

---

### Input Query 2:
"Tell me about Microsoft’s financial updates for last year."

### Output:
- "I can only assist with queries specifically related to Microsoft Ignite 2024 using the provided index. Could you adjust your query to fall within this scope?"

# Notes

- Stay strict to the Microsoft Ignite 2024 event.  
- If the query is unrelated or too ambiguous, use the out-of-scope response template.  
- Handle edge cases by erring on the side of caution and avoiding extrapolation beyond the dataset.
   
    """

    @property
    @abstractmethod
    def system_message_chat_conversation(self) -> str:
        pass

    @abstractmethod
    async def run_until_final_call(self, messages, overrides, auth_claims, should_stream) -> tuple:
        pass

    def get_system_prompt(self, override_prompt: Optional[str], follow_up_questions_prompt: str) -> str:
        if override_prompt is None:
            return self.system_message_chat_conversation.format(
                injected_prompt="", follow_up_questions_prompt=follow_up_questions_prompt
            )
        elif override_prompt.startswith(">>>"):
            return self.system_message_chat_conversation.format(
                injected_prompt=override_prompt[3:] + "\n", follow_up_questions_prompt=follow_up_questions_prompt
            )
        else:
            return override_prompt.format(follow_up_questions_prompt=follow_up_questions_prompt)

    def get_search_query(self, chat_completion: ChatCompletion, user_query: str):
        response_message = chat_completion.choices[0].message

        if response_message.tool_calls:
            for tool in response_message.tool_calls:
                if tool.type != "function":
                    continue
                function = tool.function
                if function.name == "search_sources":
                    arg = json.loads(function.arguments)
                    search_query = arg.get("search_query", self.NO_RESPONSE)
                    if search_query != self.NO_RESPONSE:
                        return search_query
        elif query_text := response_message.content:
            if query_text.strip() != self.NO_RESPONSE:
                return query_text
        return user_query

    def extract_followup_questions(self, content: Optional[str]):
        if content is None:
            return content, []
        return content.split("<<")[0], re.findall(r"<<([^>>]+)>>", content)

    async def run_without_streaming(
        self,
        messages: list[ChatCompletionMessageParam],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        session_state: Any = None,
    ) -> dict[str, Any]:
        extra_info, chat_coroutine = await self.run_until_final_call(
            messages, overrides, auth_claims, should_stream=False
        )
        chat_completion_response: ChatCompletion = await chat_coroutine
        content = chat_completion_response.choices[0].message.content
        role = chat_completion_response.choices[0].message.role
        if overrides.get("suggest_followup_questions"):
            content, followup_questions = self.extract_followup_questions(content)
            extra_info["followup_questions"] = followup_questions
        chat_app_response = {
            "message": {"content": content, "role": role},
            "context": extra_info,
            "session_state": session_state,
        }
        return chat_app_response

    async def run_with_streaming(
        self,
        messages: list[ChatCompletionMessageParam],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        session_state: Any = None,
    ) -> AsyncGenerator[dict, None]:
        extra_info, chat_coroutine = await self.run_until_final_call(
            messages, overrides, auth_claims, should_stream=True
        )
        yield {"delta": {"role": "assistant"}, "context": extra_info, "session_state": session_state}

        followup_questions_started = False
        followup_content = ""
        async for event_chunk in await chat_coroutine:
            # "2023-07-01-preview" API version has a bug where first response has empty choices
            event = event_chunk.model_dump()  # Convert pydantic model to dict
            if event["choices"]:
                completion = {
                    "delta": {
                        "content": event["choices"][0]["delta"].get("content"),
                        "role": event["choices"][0]["delta"]["role"],
                    }
                }
                # if event contains << and not >>, it is start of follow-up question, truncate
                content = completion["delta"].get("content")
                content = content or ""  # content may either not exist in delta, or explicitly be None
                if overrides.get("suggest_followup_questions") and "<<" in content:
                    followup_questions_started = True
                    earlier_content = content[: content.index("<<")]
                    if earlier_content:
                        completion["delta"]["content"] = earlier_content
                        yield completion
                    followup_content += content[content.index("<<") :]
                elif followup_questions_started:
                    followup_content += content
                else:
                    yield completion
        if followup_content:
            _, followup_questions = self.extract_followup_questions(followup_content)
            yield {"delta": {"role": "assistant"}, "context": {"followup_questions": followup_questions}}

    async def run(
        self,
        messages: list[ChatCompletionMessageParam],
        session_state: Any = None,
        context: dict[str, Any] = {},
    ) -> dict[str, Any]:
        overrides = context.get("overrides", {})
        auth_claims = context.get("auth_claims", {})
        return await self.run_without_streaming(messages, overrides, auth_claims, session_state)

    async def run_stream(
        self,
        messages: list[ChatCompletionMessageParam],
        session_state: Any = None,
        context: dict[str, Any] = {},
    ) -> AsyncGenerator[dict[str, Any], None]:
        overrides = context.get("overrides", {})
        auth_claims = context.get("auth_claims", {})
        return self.run_with_streaming(messages, overrides, auth_claims, session_state)
