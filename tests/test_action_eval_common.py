from __future__ import annotations

import unittest

from scripts import action_eval_common


class ActionEvalCommonTests(unittest.TestCase):
    def test_load_action_specs_uses_default_candidates(self) -> None:
        specs = action_eval_common.load_action_specs(labels=["fraig", "mfs"], raw_specs=[])
        self.assertEqual(specs["fraig"], "fraig")
        self.assertEqual(specs["mfs"], "renode; mfs; strash")

    def test_load_action_specs_allows_override(self) -> None:
        specs = action_eval_common.load_action_specs(
            labels=["custom"],
            raw_specs=["custom=renode; sop; custom; strash"],
        )
        self.assertEqual(specs, {"custom": "renode; sop; custom; strash"})

    def test_build_default_presets_adds_baseline_and_augmented_variants(self) -> None:
        presets = action_eval_common.build_default_presets(
            ["fraig", "dch"],
            include_combined=True,
        )
        self.assertEqual(
            presets["baseline"],
            list(action_eval_common.DEFAULT_ACTION_SPACE),
        )
        self.assertEqual(
            presets["plus_fraig"],
            [*action_eval_common.DEFAULT_ACTION_SPACE, "fraig"],
        )
        self.assertEqual(
            presets["all_selected"],
            [*action_eval_common.DEFAULT_ACTION_SPACE, "fraig", "dch"],
        )

    def test_resolve_algorithm_actions_uses_labels_for_tree_search_and_commands_for_mab(self) -> None:
        labels = ["rewrite", "fraig", "mfs"]
        self.assertEqual(
            action_eval_common.resolve_algorithm_actions("alphasyn", labels),
            labels,
        )
        self.assertEqual(
            action_eval_common.resolve_algorithm_actions("baseline_mab", labels),
            ["rewrite", "fraig", "renode; mfs; strash"],
        )


if __name__ == "__main__":
    unittest.main()
