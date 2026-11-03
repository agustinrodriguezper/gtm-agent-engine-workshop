import importlib
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch


HAS_MODEL_KEY = bool(os.environ.get("OPENAI_API_KEY"))
with patch.dict(os.environ, {
    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY") or "test-key",
    "LANGSMITH_TRACING": "false",
}), patch("dotenv.load_dotenv"):
    agent = importlib.import_module("gtm_agent.gtm_agent")


class SendProspectEmailTests(unittest.TestCase):
    def setUp(self):
        self.prospect = {
            "prospect_id": "LEAD-TEST",
            "name": "Test Prospect",
            "email": "prospect@example.com",
        }
        self.rep = {"name": "Test Rep", "email": "rep@example.com"}
        self.runtime = SimpleNamespace(config={"metadata": {"user_id": "REP-TEST"}})

    def send(self, prospect=None):
        return agent.send_prospect_email.func(
            prospect=self.prospect if prospect is None else prospect,
            subject="Follow up",
            body="Hello",
            runtime=self.runtime,
        )

    def test_disqualified_source_record_blocks_send(self):
        self.prospect["disqualified"] = False
        with patch.object(agent.data_service, "get_prospect_record", return_value={"disqualified": True}), \
             patch.object(agent.uuid, "uuid4") as message_id:
            result = self.send()
        self.assertEqual(result, {
            "status": "blocked",
            "error": "Prospect is disqualified; email not sent.",
        })
        message_id.assert_not_called()

    def test_existing_send_path_is_preserved(self):
        cases = (
            (self.prospect, {"disqualified": False}),
            (self.prospect, {}),
            ({k: v for k, v in self.prospect.items() if k != "prospect_id"}, {"disqualified": True}),
            (self.prospect, None),
        )
        for prospect, record in cases:
            with self.subTest(prospect=prospect, record=record), \
                 patch.object(agent.data_service, "get_prospect_record", return_value=record), \
                 patch.object(agent.data_service, "get_rep", return_value=self.rep):
                result = self.send(prospect)
                self.assertEqual(result["status"], "sent")
                self.assertTrue(result["message_id"].startswith("msg-"))
                self.assertEqual(result["to"], self.prospect["email"])
                self.assertEqual(result["to_name"], self.prospect["name"])
                self.assertEqual(result["from"], self.rep["email"])
                self.assertEqual(result["from_name"], self.rep["name"])
                self.assertEqual(result["subject"], "Follow up")
                self.assertEqual(result["body"], "Hello")

    def test_missing_email_still_fails(self):
        with patch.object(agent.data_service, "get_rep", return_value=self.rep):
            result = self.send({"name": "Test Prospect"})
        self.assertEqual(result, {
            "status": "failed",
            "error": "Prospect record has no email address.",
        })

    @unittest.skipUnless(HAS_MODEL_KEY, "Requires OPENAI_API_KEY for the live agent")
    def test_agent_explains_blocked_send(self):
        pid = next(pid for pid, rec in agent.data_service.PROSPECTS.items() if rec.get("disqualified"))
        with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}):
            result = agent.run_agent(
                f"Send an email to prospect {pid} with subject 'Follow up' and body 'Hello'.",
                user_id=agent.REP_IDS[0]["rep_id"],
            )
        sends = [
            json.loads(message.content)
            for message in result["messages"]
            if getattr(message, "type", None) == "tool" and message.name == "send_prospect_email"
        ]
        self.assertTrue(sends)
        self.assertTrue(all(send == {
            "status": "blocked",
            "error": "Prospect is disqualified; email not sent.",
        } for send in sends))
        reply = result["reply"].lower()
        self.assertIn("disqualified", reply)
        self.assertRegex(
            reply,
            r"no email (was |has been )?sent|"
            r"(?:not|haven't|wasn't|didn't|did not|couldn't|cannot|can't|unable to)[^.]*send|"
            r"(?:not|wasn't|hasn't been) sent",
        )


if __name__ == "__main__":
    unittest.main()
