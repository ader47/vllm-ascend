# SPDX-License-Identifier: Apache-2.0

import ast
import unittest
from pathlib import Path


class TestDSATwoNodePreflight(unittest.TestCase):
    def test_mc2_matches_runtime_and_inference(self):
        demo_dir = Path(__file__).resolve().parents[2] / "examples" / "dsa_demo"
        script = (demo_dir / "check_two_node_dp16_ep16.sh").read_text(encoding="utf-8")
        program = script.split("cat >\"$TEST_FILE\" <<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
        tree = ast.parse(program)
        mc2 = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "test_mc2")
        hidden_size = next(
            ast.literal_eval(node.value)
            for node in mc2.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "hidden_size" for target in node.targets)
        )
        # Bounds reported by the Ascend 950 MC2 tiling check; H=128 is invalid.
        self.assertGreaterEqual(hidden_size, 1024)
        self.assertLessEqual(hidden_size, 8192)
        op_names = {"npu_moe_distribute_dispatch_v2", "npu_moe_distribute_combine_v2"}
        calls = [
            node
            for node in ast.walk(mc2)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in op_names
        ]
        self.assertCountEqual([call.func.attr for call in calls], op_names)
        for call in calls:
            # Both real MC2 operations must run without overriding comm_alg.
            self.assertNotIn("comm_alg", [keyword.arg for keyword in call.keywords])
        self.assertTrue(
            any(
                isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "test_mc2"
                for node in ast.walk(tree)
            )
        )

        inference = ast.parse((demo_dir / "simple_prompt_test_dp16.py").read_text(encoding="utf-8"))
        hierarchy_values = [
            node.value.value
            for node in ast.walk(inference)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Constant)
            and any(
                isinstance(target, ast.Subscript)
                and isinstance(target.slice, ast.Constant)
                and target.slice.value == "enable_mc2_hierarchy_comm"
                for target in node.targets
            )
        ]
        self.assertEqual(hierarchy_values, [False])


if __name__ == "__main__":
    unittest.main()
