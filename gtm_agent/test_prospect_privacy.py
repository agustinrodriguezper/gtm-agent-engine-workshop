import importlib
import json
import unittest
from unittest.mock import Mock, patch

from gtm_agent import data_service

with patch("dotenv.load_dotenv"), \
        patch("langchain_openai.ChatOpenAI"), \
        patch("deepagents.create_deep_agent"):
    agent = importlib.import_module("gtm_agent.gtm_agent")


class ProspectPrivacyTests(unittest.TestCase):
    def setUp(self):
        self.cache = patch.dict(data_service._PROFILES, {}, clear=True)
        self.cache.start()
        self.addCleanup(self.cache.stop)

    def test_contact_contains_only_contact_fields(self):
        for prospect_id, record in data_service.PROSPECTS.items():
            with self.subTest(prospect_id=prospect_id):
                result = agent.get_prospect.invoke({"prospect_id": prospect_id})
                self.assertEqual(result, {
                    "prospect": {
                        "prospect_id": prospect_id,
                        **{key: record[key] for key in
                           ("name", "email", "disqualified")},
                    },
                    "found": True,
                })

    def test_optional_contact_fields_are_preserved_when_present(self):
        with patch.dict(data_service.PROSPECTS, {"minimal": {"email": None}}):
            self.assertEqual(agent.get_prospect.invoke({"prospect_id": "minimal"}), {
                "prospect": {"prospect_id": "minimal", "email": None},
                "found": True,
            })

    def test_fresh_and_cached_profiles_are_safe_for_scoring(self):
        for prospect_id, record in data_service.PROSPECTS.items():
            expected = {
                "prospect_id": prospect_id,
                **{key: record[key] for key in (
                    "name", "email", "annual_revenue", "engagement_history",
                    "account_details", "tech_stack", "enrichment_source",
                )},
            }
            for cached in (False, True):
                with self.subTest(prospect_id=prospect_id, cached=cached):
                    if cached:
                        data_service._PROFILES[prospect_id] = {
                            **expected,
                            "billing_qualification": record["billing_qualification"],
                        }
                    result = agent.build_prospect_profile.invoke({
                        "prospect_id": prospect_id,
                    })
                    self.assertEqual(result, {
                        "prospect_profile": expected, "found": True,
                    })
                    if not cached:
                        self.assertEqual(data_service._PROFILES[prospect_id], expected)
                    self.assertEqual(data_service.get_profile_from_db(prospect_id), {
                        "prospect_profile": expected,
                    })
                    arguments = {
                        "prospect_profile": result["prospect_profile"],
                        "offering": data_service.get_offering("OFFER-10001"),
                    }
                    self.assertNotIn("billing_qualification", json.dumps(arguments))
                    score = {"score": 75, "justification": "Good fit"}
                    scoring_llm = Mock()
                    scoring_llm.invoke.return_value.model_dump.return_value = score
                    with patch.object(agent, "_scoring_llm", scoring_llm):
                        self.assertEqual(agent.score_prospect.invoke(arguments), score)
                    messages = scoring_llm.invoke.call_args.args[0]
                    sent_profile = json.loads(messages[1]["content"].split(
                        "\n\nProspect profile:\n", 1
                    )[1])
                    self.assertEqual(sent_profile, expected)

    def test_missing_and_empty_profiles_keep_found_behavior(self):
        self.assertEqual(data_service.get_profile_from_db("missing"), {
            "prospect_profile": None,
        })
        self.assertEqual(agent.build_prospect_profile.invoke({
            "prospect_id": "missing",
        }), {"prospect_profile": None, "found": False})
        self.assertEqual(agent.get_prospect.invoke({"prospect_id": "missing"}), {
            "prospect": None, "found": False,
        })
        data_service._PROFILES["empty"] = {}
        self.assertEqual(agent.build_prospect_profile.invoke({
            "prospect_id": "empty",
        }), {"prospect_profile": {}, "found": True})


if __name__ == "__main__":
    unittest.main()
