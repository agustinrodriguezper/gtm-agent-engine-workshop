import json
import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Event
from unittest.mock import patch

from gtm_agent import data_service

with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    from gtm_agent import gtm_agent as agent


class ProspectUpdateTests(unittest.TestCase):
    def setUp(self):
        self.prospect_id = "LEAD-39002"
        self.prospects = deepcopy(data_service.PROSPECTS)
        self.prospects[self.prospect_id]["tech_stack"] = ["AWS"]
        self.prospect_patch = patch.object(data_service, "PROSPECTS", self.prospects)
        self.profile_patch = patch.object(data_service, "_PROFILES", {})
        self.env_patch = patch.dict(os.environ, {"LANGSMITH_TRACING": "false"})
        for patcher in (self.prospect_patch, self.profile_patch, self.env_patch):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_update_persists_and_returns_saved_stack(self):
        result = data_service.update_prospect_info(self.prospect_id, "Okta")

        self.assertEqual(result, {
            "updated": True, "found": True, "tech_stack": ["AWS", "Okta"],
        })
        self.assertEqual(self.prospects[self.prospect_id]["tech_stack"], result["tech_stack"])
        self.assertEqual(data_service.fetch_tech_stack(self.prospect_id), ["AWS", "Okta"])

    def test_update_refreshes_cached_profile(self):
        agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})

        agent.update_prospect_info.invoke({"prospect_id": self.prospect_id, "technology": "Okta"})
        result = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})

        self.assertTrue(result["found"])
        self.assertEqual(result["prospect_profile"]["tech_stack"], ["AWS", "Okta"])

    def test_in_flight_profile_build_cannot_cache_old_stack(self):
        fetched = Event()
        resume = Event()
        fetch_tech_stack = data_service.fetch_tech_stack

        def paused_fetch(prospect_id):
            stack = fetch_tech_stack(prospect_id)
            fetched.set()
            if not resume.wait(5):
                raise TimeoutError("Profile build did not resume")
            return stack

        with patch.object(data_service, "fetch_tech_stack", side_effect=paused_fetch):
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    agent.build_prospect_profile.invoke, {"prospect_id": self.prospect_id},
                )
                try:
                    self.assertTrue(fetched.wait(5))
                    data_service.update_prospect_info(self.prospect_id, "Okta")
                finally:
                    resume.set()
                built = future.result(timeout=5)

        cached = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        self.assertEqual(built["prospect_profile"]["tech_stack"], ["AWS", "Okta"])
        self.assertEqual(cached["prospect_profile"]["tech_stack"], ["AWS", "Okta"])

    def test_scoring_after_update_uses_saved_stack(self):
        profile = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        agent.update_prospect_info.invoke({"prospect_id": self.prospect_id, "technology": "Okta"})
        offering = {**data_service.get_offering("OFFER-10004"), "required_tech_stack": ["Okta"]}
        score = agent.ProspectScore(
            score=100, justification="Has Okta; no required technologies missing.",
            rubric_breakdown=agent.RubricBreakdown(
                revenue_fit=100, tech_stack_match=100, segment_fit=100,
            ),
        )

        with patch.object(agent, "_scoring_llm") as scoring_llm:
            scoring_llm.invoke.return_value = score
            result = agent.score_prospect.invoke({
                "prospect_profile": profile["prospect_profile"], "offering": offering,
            })

        content = scoring_llm.invoke.call_args.args[0][1]["content"]
        scored_profile = json.loads(content.split("\n\nProspect profile:\n", 1)[1])
        missing = set(offering["required_tech_stack"]) - set(scored_profile["tech_stack"])
        self.assertEqual(missing, set())
        self.assertEqual(result, score.model_dump())

    def test_unknown_prospect_leaves_records_and_profiles_unchanged(self):
        agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        before_records = deepcopy(self.prospects)
        before_profile = deepcopy(data_service.get_profile_from_db(self.prospect_id))

        result = data_service.update_prospect_info("UNKNOWN", "Okta")

        self.assertEqual(result, {"updated": False, "found": False})
        self.assertEqual(self.prospects, before_records)
        self.assertEqual(data_service.get_profile_from_db(self.prospect_id), before_profile)
        self.assertIsNone(data_service.get_profile_from_db("UNKNOWN")["prospect_profile"])

    def test_repeated_update_does_not_duplicate_technology(self):
        data_service.update_prospect_info(self.prospect_id, "Okta")
        data_service.update_prospect_info(self.prospect_id, "Okta")

        self.assertEqual(data_service.fetch_tech_stack(self.prospect_id), ["AWS", "Okta"])


if __name__ == "__main__":
    unittest.main()
