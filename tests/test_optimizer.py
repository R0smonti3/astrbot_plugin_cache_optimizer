from __future__ import annotations

import os
import sys
import unittest

# 确保能导入当前插件的 main 模块
current_dir = os.path.dirname(os.path.abspath(__file__))
plugin_root = os.path.dirname(current_dir)
if plugin_root not in sys.path:
    sys.path.insert(0, plugin_root)

from main import CacheOptimizerPlugin, _is_valid_user_message, _extract_usage_tokens, _check_system_prompt_timestamp


class DummyContext:
    def __init__(self, max_turns: int = -1, strategy: str = "truncate_by_turns", raw_cfg: dict | None = None):
        if raw_cfg is not None:
            self._cfg = raw_cfg
        else:
            self._cfg = {
                "agent_runner": {
                    "config": {
                        "compression": {
                            "max_turns": max_turns,
                            "overflow_strategy": strategy,
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
                "low_watermark_turns": 10,
                "step_turns": 6,
                "debug_log": False,
            },
        )

    def test_default_config_values(self):
        """场景 0：空配置时应正确返回新默认值 L=10, S=6"""
        default_plugin = CacheOptimizerPlugin(self.ctx, {})
        self.assertEqual(default_plugin._low_watermark_turns(), 10)
        self.assertEqual(default_plugin._step_turns(), 6)

    def test_continuous_100_turns_eviction_intervals(self):
        """场景 1：普通对话连续 100 轮的裁剪触发位置是否严格间隔 S 轮"""
        L = 10
        S = 6
        threshold = L + S  # 16 轮历史，当前为第 17 轮

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

        self.assertEqual(eviction_turn_records[0], 17)
        for interval in intervals:
            self.assertEqual(interval, S)

    def test_persona_preset_dialogs_retention(self):
        """场景 2：Persona 预设对话 (_no_save=True) 既排除轮次计数又在裁剪中完整保留前缀"""
        L = 10
        S = 6

        # 构造带有 2 轮 Persona 情景预设对话的历史（_no_save=True）
        persona_presets = [
            {"role": "user", "content": "你好，请扮演小助手", "_no_save": True},
            {"role": "assistant", "content": "收到，我是你的专属小助手！", "_no_save": True},
            {"role": "user", "content": "你有名字吗？", "_no_save": True},
            {"role": "assistant", "content": "我叫小艾。", "_no_save": True},
        ]

        for msg in persona_presets:
            self.assertFalse(_is_valid_user_message(msg))

        normal_history = []
        for i in range(1, 17):  # 达到 10 + 6 = 16 轮门槛
            normal_history.append({"role": "user", "content": f"normal_user_{i}"})
            normal_history.append({"role": "assistant", "content": f"normal_bot_{i}"})

        full_contexts = list(persona_presets) + normal_history
        req = DummyReq(contexts=full_contexts)

        counted_user_turns = sum(1 for m in req.contexts if _is_valid_user_message(m))
        self.assertEqual(counted_user_turns, 16)

        self.plugin._apply_chunked_eviction(req, "test_session_persona")

        self.assertEqual(len(req.contexts), 4 + L * 2)
        self.assertEqual(req.contexts[:4], persona_presets)
        self.assertEqual(req.contexts[4]["content"], f"normal_user_{16 - L + 1}")
        post_counted_user = sum(1 for m in req.contexts if _is_valid_user_message(m))
        self.assertEqual(post_counted_user, L)

    def test_multi_tool_calls_transaction_boundary(self):
        """场景 3：一轮包含多个连续 Tool Calls 复合调用时仍按 1 轮计算且切口无孤立 Tool 消息"""
        L = 10

        complex_contexts: list[dict] = []
        for turn in range(1, 25):
            complex_contexts.append({"role": "user", "content": f"complex_task_{turn}"})
            if turn % 3 == 0:
                # 恢复包含连续 2 组 Tool Call 的复杂多工具事务链路
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
        self.assertEqual(sum(1 for m in req.contexts if _is_valid_user_message(m)), 24)

        self.plugin._apply_chunked_eviction(req, "test_session_tool")

        self.assertEqual(sum(1 for m in req.contexts if _is_valid_user_message(m)), L)
        self.assertTrue(_is_valid_user_message(req.contexts[0]))
        self.assertNotIn("tool_call_id", req.contexts[0])
        self.assertEqual(req.contexts[0]["content"], f"complex_task_{24 - L + 1}")

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

    def test_framework_config_health_diagnostics(self):
        """场景 5：框架配置冲突实时诊断（三态校验：warning / healthy / unknown）"""
        # 1. 冲突状态 (warning)：max_turns=12 <= 16 轮门槛，且 strategy=llm_compress
        conflict_ctx = DummyContext(max_turns=12, strategy="llm_compress")
        plugin_conflict = CacheOptimizerPlugin(conflict_ctx, {"low_watermark_turns": 10, "step_turns": 6})
        status, warnings = plugin_conflict._diagnose_framework_config("test_umo")
        self.assertEqual(status, "warning")
        self.assertEqual(len(warnings), 2)
        self.assertTrue(any("max_turns=12" in w for w in warnings))
        self.assertTrue(any("llm_compress" in w for w in warnings))

        # 1.2 有限轮数但大于门槛 (warning)：max_turns=20 > 16 轮门槛
        finite_ctx = DummyContext(max_turns=20, strategy="truncate_by_turns")
        plugin_finite = CacheOptimizerPlugin(finite_ctx, {"low_watermark_turns": 10, "step_turns": 6})
        status_finite, warnings_finite = plugin_finite._diagnose_framework_config("test_umo")
        self.assertEqual(status_finite, "warning")
        self.assertTrue(any("max_turns=20" in w and "有限轮数" in w for w in warnings_finite))

        # 2. 健康状态 (healthy)：max_turns=-1，strategy=truncate_by_turns
        healthy_ctx = DummyContext(max_turns=-1, strategy="truncate_by_turns")
        plugin_healthy = CacheOptimizerPlugin(healthy_ctx, {"low_watermark_turns": 10, "step_turns": 6})
        status, no_warnings = plugin_healthy._diagnose_framework_config("test_umo")
        self.assertEqual(status, "healthy")
        self.assertEqual(len(no_warnings), 0)

        # 3. 未知状态 (unknown)：无法获取有效配置
        empty_ctx = DummyContext(raw_cfg={})
        plugin_empty = CacheOptimizerPlugin(empty_ctx, {"low_watermark_turns": 10, "step_turns": 6})
        status, unknown_msgs = plugin_empty._diagnose_framework_config("test_umo")
        self.assertEqual(status, "unknown")
        self.assertTrue(any("无法读取" in m or "未配置" in m for m in unknown_msgs))

    def test_extract_multi_provider_usage_tokens(self):
        """场景 6：跨厂商 Token Usage 解析兼容测试 (DeepSeek, OpenAI, Claude, Gemini, Dict)"""
        # 1. DeepSeek (input_cached, input_other)
        ds_usage = {"input_cached": 1200, "input_other": 300}
        c, o = _extract_usage_tokens(ds_usage)
        self.assertEqual((c, o), (1200, 300))

        # 2. OpenAI (prompt_tokens_details.cached_tokens)
        openai_usage = {
            "prompt_tokens": 1500,
            "prompt_tokens_details": {"cached_tokens": 1000},
        }
        c, o = _extract_usage_tokens(openai_usage)
        self.assertEqual((c, o), (1000, 500))

        # 3. Claude (cache_read_input_tokens + input_tokens)
        claude_usage = {
            "input_tokens": 2000,
            "cache_read_input_tokens": 1600,
        }
        c, o = _extract_usage_tokens(claude_usage)
        self.assertEqual((c, o), (1600, 400))

        # 4. Gemini (cached_content_token_count + prompt_token_count)
        gemini_usage = {
            "prompt_token_count": 800,
            "cached_content_token_count": 600,
        }
        c, o = _extract_usage_tokens(gemini_usage)
        self.assertEqual((c, o), (600, 200))

    def test_system_prompt_timestamp_diagnostics(self):
        """场景 7：系统提示词秒/分级高频动态时间戳击穿缓存诊断"""
        # 1. 包含动态时间戳（中文格式）
        bad_sp1 = "你是助手。当前时间：2026-11-06 21:05:00，请回答问题。"
        w1 = _check_system_prompt_timestamp(bad_sp1)
        self.assertEqual(len(w1), 1)
        self.assertTrue("击穿失效" in w1[0])

        # 2. 包含动态时间戳（英文格式）
        bad_sp2 = "You are a helpful assistant. Current time: 2026/11/06 21:05."
        w2 = _check_system_prompt_timestamp(bad_sp2)
        self.assertEqual(len(w2), 1)

        # 3. 正常静态人设（不含时间戳）
        good_sp = "你是一只猫娘，说话要带喵。"
        w3 = _check_system_prompt_timestamp(good_sp)
        self.assertEqual(len(w3), 0)

    def test_session_level_stats_isolation(self):
        """场景 8：单会话统计隔离验证"""
        import asyncio

        class DummyEventWithUmo:
            def __init__(self, umo):
                self.unified_msg_origin = umo

        e1 = DummyEventWithUmo("session_group_1")
        e2 = DummyEventWithUmo("session_group_2")

        r1 = DummyResp(cached=1000, other=200)
        r2 = DummyResp(cached=500, other=500)

        asyncio.run(self.plugin.on_llm_response(e1, r1))
        asyncio.run(self.plugin.on_llm_response(e2, r2))

        s1 = self.plugin._get_session_stat("session_group_1")
        s2 = self.plugin._get_session_stat("session_group_2")

        self.assertEqual(s1["cached_tokens"], 1000)
        self.assertEqual(s1["other_tokens"], 200)
        self.assertEqual(s2["cached_tokens"], 500)
        self.assertEqual(s2["other_tokens"], 500)

        # 全局总计仍然正确累加
        self.assertEqual(self.plugin._stats["total_cached_tokens"], 1500)
        self.assertEqual(self.plugin._stats["total_other_tokens"], 700)

    def test_eviction_timestamp_and_status_payload(self):
        """场景 9：上一轮剪切时间记录、Web看板数据导出及 /cache_opt 指令回复校验"""
        import asyncio
        from main import _format_eviction_time

        # 1. 初始状态未发生剪切
        self.assertEqual(_format_eviction_time(0), "暂无剪切记录")
        init_payload = self.plugin._status_payload()
        self.assertEqual(init_payload["last_eviction_ts"], 0.0)
        self.assertEqual(len(init_payload["eviction_logs"]), 0)

        # 2. 构造 16 轮对话触发一次裁剪 (L=10, S=6)
        contexts = []
        for i in range(16):
            contexts.append({"role": "user", "content": f"u{i}"})
            contexts.append({"role": "assistant", "content": f"a{i}"})

        req = DummyReq(contexts=contexts)
        self.plugin._apply_chunked_eviction(req, "qq:GroupMessage:123456")

        payload = self.plugin._status_payload()
        self.assertGreater(payload["last_eviction_ts"], 0)
        self.assertEqual(payload["last_eviction_turns"], 6)
        self.assertEqual(payload["last_eviction_umo"], "qq:GroupMessage:123456")
        self.assertEqual(len(payload["eviction_logs"]), 1)
        self.assertEqual(payload["sessions"][0]["current_user_turns"], 10)
        self.assertGreater(payload["sessions"][0]["last_eviction_ts"], 0)

        # 3. 测试 /cache_opt 指令回复包含“上一轮剪切时间”
        class DummyCmdEvent:
            def __init__(self):
                self.unified_msg_origin = "qq:GroupMessage:123456"
                self.message_str = "/cache_opt"
                self.results = []
            def is_admin(self):
                return True
            def plain_result(self, text):
                return text

        ev = DummyCmdEvent()
        async def run_cmd():
            out = []
            async for r in self.plugin.cmd_cache_opt(ev):
                out.append(r)
            return out

        outputs = asyncio.run(run_cmd())
        self.assertEqual(len(outputs), 1)
        self.assertIn("上一轮剪切时间:", outputs[0])
        self.assertIn("裁掉 6 轮", outputs[0])


if __name__ == "__main__":
    unittest.main()
