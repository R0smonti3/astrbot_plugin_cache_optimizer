from __future__ import annotations

import os
import sys
import unittest

# 确保能导入当前插件的 main 模块
current_dir = os.path.dirname(os.path.abspath(__file__))
plugin_root = os.path.dirname(current_dir)
if plugin_root not in sys.path:
    sys.path.insert(0, plugin_root)

from main import CacheOptimizerPlugin, _is_valid_user_message


class DummyContext:
    def __init__(self):
        self._cfg = {
            "agent_runner": {
                "config": {
                    "compression": {
                        "max_turns": -1,
                        "overflow_strategy": "truncate_by_turns",
                    }
                }
            }
        }

    def get_config(self, umo=None):
        return self._cfg


class DummyReq:
    def __init__(self, contexts: list[dict]):
        self.contexts = contexts


class DummyResp:
    def __init__(self, cached: int, other: int):
        class Usage:
            def __init__(self, c, o):
                self.input_cached = c
                self.input_other = o

        self.usage = Usage(cached, other)


class TestCacheOptimizer(unittest.TestCase):
    def setUp(self):
        self.ctx = DummyContext()
        self.plugin = CacheOptimizerPlugin(
            self.ctx,
            {
                "enabled": True,
                "low_watermark_turns": 18,
                "step_turns": 12,
                "debug_log": False,
            },
        )

    def test_continuous_100_turns_eviction_intervals(self):
        """场景 1：普通对话连续 100 轮的裁剪触发位置是否严格间隔 S 轮"""
        L = 18
        S = 12
        threshold = L + S  # 30 轮历史，当前为第 31 轮

        history: list[dict] = []
        eviction_turn_records: list[int] = []

        for turn in range(1, 101):
            req = DummyReq(contexts=list(history))
            prior_user_turns = sum(1 for m in req.contexts if _is_valid_user_message(m))

            self.plugin._apply_chunked_eviction(req, "test_session")

            post_user_turns = sum(1 for m in req.contexts if _is_valid_user_message(m))
            if post_user_turns < prior_user_turns:
                eviction_turn_records.append(turn)
                self.assertEqual(prior_user_turns, threshold)
                self.assertEqual(post_user_turns, L)
                self.assertTrue(_is_valid_user_message(req.contexts[0]))

            history = req.contexts
            history.append({"role": "user", "content": f"user_query_{turn}"})
            history.append({"role": "assistant", "content": f"bot_reply_{turn}"})

        intervals = [
            eviction_turn_records[i] - eviction_turn_records[i - 1]
            for i in range(1, len(eviction_turn_records))
        ]

        self.assertEqual(eviction_turn_records[0], 31)
        for interval in intervals:
            self.assertEqual(interval, S)

    def test_persona_preset_dialogs_retention(self):
        """场景 2：Persona 预设对话 (_no_save=True) 既排除轮次计数又在裁剪中完整保留前缀"""
        L = 18
        S = 12

        # 构造带有 2 轮 Persona 情景预设对话的历史（_no_save=True）
        persona_presets = [
            {"role": "user", "content": "你好，请扮演小助手", "_no_save": True},
            {"role": "assistant", "content": "收到，我是你的专属小助手！", "_no_save": True},
            {"role": "user", "content": "你有名字吗？", "_no_save": True},
            {"role": "assistant", "content": "我叫小艾。", "_no_save": True},
        ]

        # 1. 验证计数排除：预设对话不被误算为普通 User 轮数
        for msg in persona_presets:
            self.assertFalse(_is_valid_user_message(msg))

        # 2. 构造 30 轮普通历史（达到 18 + 12 = 30 轮阈值）
        normal_history = []
        for i in range(1, 31):
            normal_history.append({"role": "user", "content": f"normal_user_{i}"})
            normal_history.append({"role": "assistant", "content": f"normal_bot_{i}"})

        # 总上下文 = 预设前缀 (4条) + 普通历史 (60条)
        full_contexts = list(persona_presets) + normal_history
        req = DummyReq(contexts=full_contexts)

        counted_user_turns = sum(1 for m in req.contexts if _is_valid_user_message(m))
        self.assertEqual(counted_user_turns, 30)

        # 3. 执行裁剪
        self.plugin._apply_chunked_eviction(req, "test_session_persona")

        # 4. 验证保留：预设对话 4 条完整保留在最前端
        self.assertEqual(len(req.contexts), 4 + L * 2)
        self.assertEqual(req.contexts[:4], persona_presets)

        # 5. 验证切口首条普通消息是倒数第 18 轮（即 normal_user_13）
        self.assertEqual(req.contexts[4]["content"], "normal_user_13")
        post_counted_user = sum(1 for m in req.contexts if _is_valid_user_message(m))
        self.assertEqual(post_counted_user, L)

    def test_multi_tool_calls_transaction_boundary(self):
        """场景 3：一轮包含多个 Tool Calls 时仍按 1 轮计算且切口无孤立 Tool 消息"""
        L = 18

        complex_contexts: list[dict] = []
        for turn in range(1, 36):
            complex_contexts.append({"role": "user", "content": f"complex_task_{turn}"})
            if turn % 3 == 0:
                complex_contexts.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": f"call_{turn}_1", "function": {"name": "query_db"}}],
                })
                complex_contexts.append({
                    "role": "tool",
                    "tool_call_id": f"call_{turn}_1",
                    "content": f"result_{turn}_1",
                })
                complex_contexts.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": f"call_{turn}_2", "function": {"name": "calc"}}],
                })
                complex_contexts.append({
                    "role": "tool",
                    "tool_call_id": f"call_{turn}_2",
                    "content": f"result_{turn}_2",
                })
                complex_contexts.append({
                    "role": "assistant",
                    "content": f"final_summary_{turn}",
                })
            else:
                complex_contexts.append({"role": "assistant", "content": f"simple_reply_{turn}"})

        req = DummyReq(contexts=complex_contexts)
        self.assertEqual(sum(1 for m in req.contexts if _is_valid_user_message(m)), 35)

        self.plugin._apply_chunked_eviction(req, "test_session_tool")

        self.assertEqual(sum(1 for m in req.contexts if _is_valid_user_message(m)), L)
        self.assertTrue(_is_valid_user_message(req.contexts[0]))
        self.assertNotIn("tool_call_id", req.contexts[0])
        self.assertEqual(req.contexts[0]["content"], f"complex_task_{35 - L + 1}")

    def test_real_usage_token_statistics(self):
        """场景 4：真实的 Token usage 统计与加权命中率准确度"""
        import asyncio

        class DummyEvent:
            pass

        event = DummyEvent()
        resp1 = DummyResp(cached=3000, other=1000)
        resp2 = DummyResp(cached=2500, other=500)

        asyncio.run(self.plugin.on_llm_response(event, resp1))
        asyncio.run(self.plugin.on_llm_response(event, resp2))

        self.assertEqual(self.plugin._stats["total_cached_tokens"], 5500)
        self.assertEqual(self.plugin._stats["total_other_tokens"], 1500)
        self.assertEqual(self.plugin._stats["response_count"], 2)


if __name__ == "__main__":
    unittest.main()
