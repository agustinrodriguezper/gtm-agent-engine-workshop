"""Offline tests for signed-in rep context and tool execution."""

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from pydantic import Field

with patch.dict(os.environ, {"OPENAI_API_KEY": "test", "LANGSMITH_TRACING": "false"}):
    from gtm_agent import gtm_agent as agent


class RecordingModel(GenericFakeChatModel):
    requests: list = Field(default_factory=list)

    def _generate(self, messages, **kwargs):
        self.requests.append(messages)
        return super()._generate(messages, **kwargs)


class RepContextTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {"LANGSMITH_TRACING": "false"})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_configured_rep_is_available_before_invocation_for_each_intent(self):
        requests = [
            "Email LEAD-12853 to schedule a demo.",
            "Add Kafka to LEAD-71001, then score against OFFER-10005.",
            "Score LEAD-71001 against OFFER-10005.",
        ]
        for rep in agent.REP_IDS:
            for request in requests:
                with self.subTest(rep=rep["rep_id"], request=request):
                    graph = Mock()
                    graph.invoke.return_value = {"messages": [AIMessage(content="Done")]}
                    with patch.object(agent, "gtm_agent", graph):
                        result = agent.run_agent(request, user_id=rep["rep_id"])
                    inputs = graph.invoke.call_args.args[0]
                    config = graph.invoke.call_args.kwargs["config"]
                    context, user = inputs["messages"]
                    self.assertEqual(context["role"], "system")
                    self.assertIn(json.dumps(rep), context["content"])
                    self.assertEqual(user, {"role": "user", "content": request})
                    self.assertEqual(config["metadata"]["user_id"], rep["rep_id"])
                    self.assertEqual(result["reply"], "Done")

    def test_randomly_selected_rep_is_also_injected(self):
        rep = agent.REP_IDS[-1]
        graph = Mock()
        graph.invoke.return_value = {"messages": [AIMessage(content="Done")]}
        with patch.object(agent.random, "choice", return_value=rep):
            with patch.object(agent, "gtm_agent", graph):
                agent.run_agent("Email LEAD-12853.")
        context = graph.invoke.call_args.args[0]["messages"][0]
        self.assertIn(json.dumps(rep), context["content"])

    def test_email_graph_can_lookup_and_send_with_injected_signoff(self):
        rep = agent.REP_IDS[0]
        model = RecordingModel(messages=iter(()))

        def responses():
            context = next(
                message.content for message in model.requests[0]
                if isinstance(message, SystemMessage) and "Signed-in rep: " in message.content
            )
            sender = json.loads(context.split("Signed-in rep: ", 1)[1].split("\n", 1)[0])
            yield AIMessage(content="", tool_calls=[{
                "name": "get_prospect", "args": {"prospect_id": "LEAD-12853"}, "id": "lookup",
            }])
            contact = json.loads(model.requests[-1][-1].content)["prospect"]
            yield AIMessage(content="", tool_calls=[{
                "name": "send_prospect_email",
                "args": {"prospect": contact, "subject": "Demo", "body": "Let's meet.\n" + sender["name"]},
                "id": "send",
            }])
            yield AIMessage(content="Sent")

        model.messages = responses()
        with patch.object(ChatOpenAI, "bind_tools", return_value=model):
            result = agent.run_agent("Email LEAD-12853 to schedule a demo.", user_id=rep["rep_id"])
        tool_messages = [message for message in result["messages"] if isinstance(message, ToolMessage)]
        self.assertEqual([message.name for message in tool_messages], ["get_prospect", "send_prospect_email"])
        sent = json.loads(tool_messages[-1].content)
        self.assertEqual(sent["status"], "sent")
        self.assertEqual(sent["from"], rep["email"])
        self.assertEqual(sent["from_name"], rep["name"])
        self.assertTrue(sent["body"].endswith(rep["name"]))
        self.assertEqual(len(model.requests), 3)

    def test_scoring_graph_can_run_independent_lookups_without_rep_tool(self):
        model = RecordingModel(messages=iter(()))

        def responses():
            yield AIMessage(content="", tool_calls=[
                {"name": "lookup_offering", "args": {"offering_id": "OFFER-10005"}, "id": "offering"},
                {"name": "build_prospect_profile", "args": {"prospect_id": "LEAD-71001"}, "id": "profile"},
            ])
            results = {
                message.name: json.loads(message.content)
                for message in model.requests[-1] if isinstance(message, ToolMessage)
            }
            yield AIMessage(content="", tool_calls=[{
                "name": "score_prospect",
                "args": {
                    "offering": results["lookup_offering"]["offering"],
                    "prospect_profile": results["build_prospect_profile"]["prospect_profile"],
                },
                "id": "score",
            }])
            yield AIMessage(content="Scored")

        model.messages = responses()
        scoring = Mock()
        scoring.invoke.return_value.model_dump.return_value = {"score": 75}
        with patch.object(ChatOpenAI, "bind_tools", return_value=model):
            with patch.object(agent, "_scoring_llm", scoring):
                with patch.dict(agent.data_service._PROFILES, clear=True):
                    result = agent.run_agent(
                        "Score LEAD-71001 against OFFER-10005.", user_id="rep_jchen",
                    )
        tool_names = [message.name for message in result["messages"] if isinstance(message, ToolMessage)]
        self.assertCountEqual(tool_names, ["lookup_offering", "build_prospect_profile", "score_prospect"])
        self.assertEqual(len(model.requests), 3)
        self.assertEqual(result["reply"], "Scored")

    def test_fallback_tool_remains_available_without_rep_context(self):
        rep = agent.REP_IDS[0]
        model = RecordingModel(messages=iter([
            AIMessage(content="", tool_calls=[{"name": "get_current_rep", "args": {}, "id": "rep"}]),
            AIMessage(content="Rep found"),
        ]))
        with patch.object(ChatOpenAI, "bind_tools", return_value=model):
            result = agent.gtm_agent.invoke(
                {"messages": [{"role": "user", "content": "Who is the signed-in rep?"}]},
                config={"metadata": {"user_id": rep["rep_id"]}},
            )
        tool_result = next(message for message in result["messages"] if isinstance(message, ToolMessage))
        self.assertEqual(json.loads(tool_result.content), {"rep": rep, "found": True})

    def test_unknown_rep_does_not_inject_fabricated_context(self):
        graph = Mock()
        graph.invoke.return_value = {"messages": [AIMessage(content="Done")]}
        with patch.object(agent, "gtm_agent", graph):
            agent.run_agent("Score a prospect.", user_id="unknown")
        self.assertEqual(graph.invoke.call_args.args[0]["messages"], [
            {"role": "user", "content": "Score a prospect."},
        ])

    def test_email_sender_still_defaults_to_runtime_rep(self):
        for rep in agent.REP_IDS:
            with self.subTest(rep=rep["rep_id"]):
                result = agent.send_prospect_email.func(
                    prospect={"name": "Prospect", "email": "prospect@example.com"},
                    subject="Demo", body="Hello",
                    runtime=SimpleNamespace(config={"metadata": {"user_id": rep["rep_id"]}}),
                )
                self.assertEqual(result["from"], rep["email"])
                self.assertEqual(result["from_name"], rep["name"])
                self.assertEqual(result["body"], "Hello")


if __name__ == "__main__":
    unittest.main()
