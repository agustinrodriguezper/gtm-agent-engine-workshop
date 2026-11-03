"""Offline agent workflow tests; run with python -m unittest gtm_agent.test_gtm_agent."""

import importlib
import json
import os
import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI
import tiktoken
from requests.exceptions import RequestException


class AgentWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict(os.environ, OPENAI_API_KEY="test", LANGSMITH_TRACING="false"):
            cls.app = importlib.import_module("gtm_agent.gtm_agent")

    def setUp(self):
        self.tracing = patch.dict(os.environ, LANGSMITH_TRACING="false")
        self.tracing.start()
        self.addCleanup(self.tracing.stop)
        self.profiles = patch.dict(self.app.data_service._PROFILES, clear=True)
        self.profiles.start()
        self.addCleanup(self.profiles.stop)

    def tool_call(self, name, **args):
        return AIMessage(content="", tool_calls=[{
            "name": name, "args": args, "id": name,
        }])

    def invoke(self, request, responses):
        results = [ChatResult(generations=[ChatGeneration(message=message)])
                   for message in responses + [AIMessage(content="Done")]]
        with patch.object(ChatOpenAI, "_generate", autospec=True, side_effect=results) as model:
            result = self.app.run_agent(
                request, user_id="rep_jchen", environment="test", thread_id="workflow-test",
            )
        self.assertEqual(result["reply"], "Done")
        self.assertTrue(result["run_id"])
        tool_messages = [message for message in result["messages"]
                         if isinstance(message, ToolMessage)]
        self.assertTrue(tool_messages)
        self.assertTrue(all(message.status == "success" for message in tool_messages))
        first_call = model.call_args_list[0]
        messages = first_call.args[1]
        tools = first_call.kwargs["tools"]
        self.assertEqual(messages[0].content, self.app.SYSTEM_PROMPT)
        self.assertEqual({tool["function"]["name"] for tool in tools}, {
            "lookup_offering", "build_prospect_profile", "get_prospect",
            "send_prospect_email", "score_prospect", "update_prospect_info",
            "get_current_rep",
        })
        for tool in tools:
            self.assertNotIn("runtime", tool["function"]["parameters"].get("properties", {}))
        self.first_call = (messages, tools)
        return {message.name: json.loads(message.content) for message in tool_messages}

    def test_first_call_token_budget(self):
        self.invoke("Who is the signed-in rep?", [self.tool_call("get_current_rep")])
        messages, tools = self.first_call
        try:
            encoding = tiktoken.encoding_for_model(self.app.MODEL_NAME)
        except (OSError, RequestException) as error:
            self.skipTest(f"Tokenizer data unavailable: {error}")
        token_count = self.app.agent_model.get_num_tokens_from_messages(messages)
        token_count += len(encoding.encode(json.dumps(tools)))
        self.assertLess(token_count, 3000)
        print(f"Estimated first-call tokens including tool schemas: {token_count}")

    def test_current_rep_uses_runtime_metadata(self):
        outputs = self.invoke("Who is the signed-in rep?", [
            self.tool_call("get_current_rep"),
        ])
        self.assertEqual(outputs["get_current_rep"], {
            "rep": self.app.data_service.get_rep("rep_jchen"), "found": True,
        })

    def test_email_defaults_to_runtime_sender(self):
        prospect = {"name": "Test Prospect", "email": "prospect@example.com"}
        outputs = self.invoke("Email the prospect to schedule a demo.", [
            self.tool_call("get_current_rep"),
            self.tool_call("send_prospect_email", prospect=prospect,
                           subject="Demo", body="Please schedule a demo."),
        ])
        email = outputs["send_prospect_email"]
        self.assertEqual(email["status"], "sent")
        self.assertEqual(email["to"], prospect["email"])
        self.assertEqual(email["from"], self.app.data_service.get_rep("rep_jchen")["email"])
        self.assertEqual(email["subject"], "Demo")
        self.assertEqual(email["body"], "Please schedule a demo.")

    def test_update_then_score_workflow(self):
        prospect_id = "LEAD-71001"
        offering = self.app.data_service.get_offering("OFFER-10005")
        profile = {"prospect_id": prospect_id,
                   **self.app.data_service.get_prospect_record(prospect_id)}
        score = self.app.ProspectScore(
            score=75, justification="Test assessment",
            rubric_breakdown=self.app.RubricBreakdown(
                revenue_fit=90, tech_stack_match=60, segment_fit=75,
            ),
        )
        scoring_model = Mock()
        scoring_model.invoke.return_value = score
        with patch.object(self.app, "_scoring_llm", scoring_model):
            outputs = self.invoke("Add Kafka to LEAD-71001 then score against OFFER-10005.", [
                self.tool_call("get_current_rep"),
                self.tool_call("update_prospect_info", prospect_id=prospect_id, technology="Kafka"),
                self.tool_call("build_prospect_profile", prospect_id=prospect_id),
                self.tool_call("lookup_offering", offering_id="OFFER-10005"),
                self.tool_call("score_prospect", prospect_profile=profile, offering=offering),
            ])
        self.assertTrue(outputs["update_prospect_info"]["updated"])
        self.assertTrue(outputs["build_prospect_profile"]["found"])
        self.assertEqual(outputs["lookup_offering"]["offering"], offering)
        self.assertEqual(outputs["score_prospect"], score.model_dump())
        self.assertIn(prospect_id, self.app.data_service._PROFILES)


if __name__ == "__main__":
    unittest.main()
