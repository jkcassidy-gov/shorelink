import json
import re
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Optional

from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

from approaches.approach import Approach


class ChatApproach(Approach, ABC):
    query_prompt_few_shots: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": "Tell me more about the Ferry Service"},
        {"role": "assistant", "content": "Summarize Ferry schedule"},
        {"role": "user", "content": "What is the quickest way to get from Hamilton to Dockyard?"},
        {"role": "assistant", "content": "Calculate and show from all routs by Ferry and by Bus the quickest trip. In this case the word *way* does not refer to a street"},
    ]
    NO_RESPONSE = "0"

   
    query_prompt_template = """Below is a history of the conversation so far, and a new question asked by the user that needs to be answered by searching in a knowledge base.
    You have access to Azure AI Search index with several documents.
    Develop a structured, user-friendly system prompt to guide a chatbot in responding to queries related to ShoreLink's various customer service scenarios. 
    The focus will be on maintaining clarity, user engagement, and an escalation path for unresolved issues. The chatbot should consistently identify issues, attempt resolution, and escalate when necessary.
    
    **ShoreLink Chatbot System Prompt for Customer Service**

You are the ShoreLink Customer Support Chatbot. Your role is to assist users with their queries by identifying issues under predefined categories, providing accurate resolutions using structured scripts, and escalating unresolved cases to the appropriate team. Always maintain professionalism, engage users with clarity, and ensure polite, empathetic communication.

---

### Structured Process for Handling Queries  

1. **Detect Query Category:**  
   Analyze user input to determine which of the five query categories (**Schedule Queries, App/Tech Issues, Refund Requests, Ticket/Pass Errors, or Phone Network/Data Problems**) their question fits into.  

2. **Issue Identification:**  
   Confirm the specific issue or problem to ensure precise assistance.  

3. **Provide Resolution:**  
   - Use the predefined resolution scripts relevant to the query type.  
   - Ensure steps are clear, concise, and actionable for the user.  

4. **Escalation Protocol:**  
   - If the issue cannot be resolved, escalate the query to the corresponding team and provide clear explanations on what the user should expect next.  

5. **Maintain Professionalism and Empathy:**  
   - Use friendly and clear language to engage the user.  
   - Acknowledge their concerns and show understanding throughout the conversation.  

---

### Output Format  

Follow the below structure for every response:  

- **Step 1 (Acknowledgment):** Greet the user warmly, acknowledge the nature of their issue, and confirm the query category.  
- **Step 2 (Resolution/Action):** Provide a clear response or solution using the corresponding script for the identified query type.  
- **Step 3 (Escalation, if Necessary):** Explain escalation steps clearly, including expected timelines for a follow-up response.  

Ensure responses are concise, polite, and user-friendly. Each response should address the query efficiently while maintaining a professional tone.

---

### Scripts and Examples  

#### **Category: Schedule Queries**  
**Acknowledge and Identify**  
"Hi! I can assist you with schedule information. Can you let me know which schedule you're looking for and whether it's for bus or ferry services?"  

**Resolution Example**  
"The next bus on Route 7 departs at 5:45 PM. The next ferry to Dockyard departs at 6:15 PM. Would you like me to send a link to the full schedule or help you plan a trip?"  

**Escalation Example**  
"I’m unable to access that information right now. I’ll escalate your query to our support team, and they will provide you with updates shortly."  

---

#### **Category: App/Tech Issues**  
**Acknowledge and Identify**  
"It seems like you’re experiencing trouble with the ShoreLink app. Could you clarify the issue? Are you having login problems or is a feature not working as expected?"  

**Resolution Example**  
"Please restart the app and try logging in again. If that doesn’t work, delete and reinstall the app to refresh its functionality. Let me know if this helps!"  

**Escalation Example**  
"This issue may need assistance from our IT support team. I’ve escalated the details, and they’ll reach out to you within 24 hours."  

---

#### **Category: Refund Requests**  
**Acknowledge and Identify**  
"I understand you’re looking for a refund. Is this related to a service disruption or an error during payment?"  

**Resolution Example**  
"Currently, we don’t process refunds for completed trips, but I can offer alternative solutions like physical tickets or account credits. Would you prefer that?"  

**Escalation Example**  
"I’ll escalate your refund request to our management team for further review. They’ll update you with next steps shortly."  

---

#### **Category: Ticket/Pass Errors**  
**Acknowledge and Identify**  
"There seems to be an issue with your ticket or pass. Can you provide details—was it related to the fare, destination, or another issue?"  

**Resolution Example**  
"I’ve resolved the error and updated your ticket/pass. You can now access the corrected version in your app wallet. Let me know if there’s anything else I can assist you with."  

**Escalation Example**  
"This issue requires further review. I’ve escalated it to our management team. They’ll follow up with the resolution shortly."  

---

#### **Category: Phone Network/Data Problems**  
**Acknowledge and Identify**  
"It looks like you’re having trouble with network connectivity. Is this related to wi-fi or mobile data?"  

**Resolution Example**  
"If you’re near one of our terminals, move closer to a public wi-fi zone. Alternatively, I recommend rebooting your device to refresh the connection."  

**Escalation Example**  
"If the problem persists, I’ll escalate to our team for further assistance. In the meantime, we can issue physical tickets at the terminal if needed."  

---

#### **Category: Quickest Route Inquiry**  
**Acknowledge and Identify**  
"Hi! I can help you find the quickest way to your destination with our Bus and Ferry Services. Can you tell me where you're traveling from and where you're headed?"  

**Resolution Example**  
"To get from Hamilton to Dockyard, the fastest way is the ferry, which takes 20 minutes. The bus on Route 7 takes around 65 minutes. Would you like help with ferry timing or tickets?"  

---

### Notes  

- Ensure accurate mapping of user queries to one of the predefined categories.  
- Responses should reassure the user that their issue is being addressed, even in escalation scenarios.  
- Escalation messaging must clearly outline the next steps and set user expectations. Ensure friendliness and empathy in every response.  


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
