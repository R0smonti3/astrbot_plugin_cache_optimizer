from __future__ import annotations

import copy
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star


def _is_valid_user_message(msg: Any) -> bool:
    """检查是否为合法的普通 User 消息边界（排除 tool_call_id 与 _no_save 预设对话等特殊结构）。"""
    if not isinstance(msg, dict):
        return False
    if msg.get("role") != "user":
        return False
    if msg.get("tool_call_id"):
        return False
    if msg.get("_no_save"):
        return False
    return True


class CacheOptimizerPlugin(Star):
    """大模型提示词缓存保活插件 (CacheOptimizer)：
    1. 阶梯式历史裁剪：按合法的普通 User 轮数精准计数，当历史累计满 L + S 轮时触发裁剪，
       一次性修剪至保留最近 L 轮历史，裁剪结果通过 AstrBot 正常 Agent 保存流程持久化写回数据库；
    2. 严格事务边界对齐：裁剪起点锁定在完整 User 消息边界，完整保留其后附属的 Assistant 与 Tool 调用链路，
       杜绝工具调用事务被截断导致的 400 报错；
    3. 真实 Token 账单监控：监听 on_llm_response，直接读取 DeepSeek 服务端返回的真实 input_cached 与 input_other，
       统计最终 LLM 请求的真实加权缓存命中率；
    4. 框架配置自检警告：检测会话运行时 max_turns 配置，提醒用户将原生 max_turns 设为 -1，防止原生滑动窗口提前淘汰历史。
    """

    def __init__(self, context: Context, config: dict | None = None) -> None:
        super().__init__(context)
        self.config = config if config is not None else {}
        self._checked_umos: set[str] = set()
        self._stats = {
            "total_requests": 0,
            "eviction_count": 0,
            "evicted_user_turns": 0,
            "total_cached_tokens": 0,
            "total_other_tokens": 0,
            "response_count": 0,
        }

    # ------------------------------------------------------------------
    # 配置读取辅助
    # ------------------------------------------------------------------
    def _is_enabled(self) -> bool:
        return bool(self.config.get("enabled", True))

    def _low_watermark_turns(self) -> int:
        return max(1, int(self.config.get("low_watermark_turns", 10)))

    def _step_turns(self) -> int:
        return max(1, int(self.config.get("step_turns", 6)))

    def _debug_log_enabled(self) -> bool:
        return bool(self.config.get("debug_log", False))

    # ------------------------------------------------------------------
    # 框架原生配置诊断（支持实时检查与会话按需查询）
    # ------------------------------------------------------------------
    def _diagnose_framework_config(self, umo: str | None = None) -> tuple[str, list[str]]:
        """实时检查指定会话的 AstrBot 框架配置冲突并返回 (状态, 描述列表)。
        状态返回：
        - "warning": 存在配置冲突风险
        - "healthy": 成功读取且无配置冲突
        - "unknown": 无法读取当前会话配置
        """
        try:
            cfg = self.context.get_config(umo=umo)
            if not cfg or not isinstance(cfg, dict):
                return "unknown", ["无法读取当前会话的框架配置，暂时无法确认是否存在冲突。"]

            agent_runner = cfg.get("agent_runner")
            if not isinstance(agent_runner, dict):
                return "unknown", ["当前会话未配置 agent_runner，暂时无法确认是否存在冲突。"]
            runner_config = agent_runner.get("config")
            if not isinstance(runner_config, dict):
                return "unknown", ["当前会话 agent_runner 缺少有效 config，暂时无法确认是否存在冲突。"]
            compression = runner_config.get("compression")
            if not isinstance(compression, dict):
                return "unknown", ["当前会话未启用或未定义 compression 上下文配置。"]

            warnings: list[str] = []
            max_turns = compression.get("max_turns", -1)
            threshold_turns = self._low_watermark_turns() + self._step_turns()

            if max_turns > 0:
                if max_turns <= threshold_turns:
                    warnings.append(
                        f"原生 max_turns={max_turns} <= 插件门槛({threshold_turns}轮)。"
                        f"框架会在插件触发前抢先淘汰历史，强烈建议在 WebUI 会话设置中将 max_turns 设为 -1 (不限轮数)。"
                    )
                else:
                    warnings.append(
                        f"原生 max_turns={max_turns} (有限轮数配置)。"
                        f"AstrBot 原生截断按非系统消息总数折半折算轮次(len//2)，当工具调用产生多条消息时可能抢先触发截断破坏前缀缓存。建议在 WebUI 会话设置中将 max_turns 设为 -1 (不限轮数)。"
                    )

            strategy = compression.get("overflow_strategy", "")
            if strategy == "llm_compress":
                warnings.append(
                    "原生溢出策略为 'llm_compress' (LLM摘要压缩)。"
                    "发生上下文 Token 溢出时，框架可能调用模型压缩并重写历史，影响前缀缓存连续性。建议在 WebUI 会话设置中调整为 'truncate_by_turns'。"
                )

            if warnings:
                return "warning", warnings
            return "healthy", []
        except Exception as exc:
            logger.debug(f"[CacheOptimizer] 框架配置检测跳过: {exc}")
            return "unknown", [f"读取会话框架配置发生异常 ({exc})，暂时无法确认冲突。"]

    def _check_framework_config(self, umo: str | None = None) -> None:
        umo_key = str(umo or "global")
        if umo_key in self._checked_umos:
            return
        self._checked_umos.add(umo_key)

        status, warnings = self._diagnose_framework_config(umo)
        if status == "warning":
            for w in warnings:
                logger.warning(f"[CacheOptimizer] [警告] {w}")

    # ------------------------------------------------------------------
    # 核心拦截入口：发车前历史分块裁剪 (priority=-9999)
    # ------------------------------------------------------------------
    @filter.on_llm_request(priority=-9999)
    async def on_llm_request(self, event: AstrMessageEvent, req: Any) -> None:
        if not self._is_enabled() or not req:
            return

        self._stats["total_requests"] += 1
        umo = event.unified_msg_origin
        self._check_framework_config(umo)

        # 核心：阶梯式历史裁剪（随 AstrBot 正常流程写回数据库）
        self._apply_chunked_eviction(req, umo)

    # ------------------------------------------------------------------
    # 阶梯式历史裁剪（以普通 User 轮数精准计数，安全对齐）
    # ------------------------------------------------------------------
    def _apply_chunked_eviction(self, req: Any, umo: str) -> None:
        contexts = getattr(req, "contexts", None)
        if not contexts or not isinstance(contexts, list):
            return

        L = self._low_watermark_turns()
        S = self._step_turns()

        # 1. 扫描所有合法的普通 User 消息索引（排查 tool_call_id 与特殊结构）
        user_indices = [
            i for i, msg in enumerate(contexts) if _is_valid_user_message(msg)
        ]
        total_user_turns = len(user_indices)

        # 2. 当前钩子执行时，当前用户消息尚未加入 req.contexts。
        # 当历史中的合法 User 消息数 >= L + S 时触发裁剪。
        if total_user_turns < L + S:
            return

        # 3. 定位到倒数第 L 条合法 User 消息的索引
        cut_idx = user_indices[-L]

        # 4. 安全边界检查：确保 cut_idx 合法且指向完整普通 User 消息
        if not (0 <= cut_idx < len(contexts)):
            logger.warning(
                f"[CacheOptimizer] [警告] 裁剪索引越界 ({cut_idx}/{len(contexts)})，放弃本次裁剪以防上下文损坏。"
            )
            return

        target_msg = contexts[cut_idx]
        if not _is_valid_user_message(target_msg):
            logger.warning(
                f"[CacheOptimizer] [警告] 目标裁剪点消息非合法 User 角色 (role={target_msg.get('role')})，放弃本次裁剪。"
            )
            return

        # 5. 提取开头连续的 Persona 预设对话 (_no_save=True)，确保当轮请求人设完整且缓存前缀稳定
        prefix_no_save: list[dict] = []
        for msg in contexts:
            if isinstance(msg, dict) and msg.get("_no_save"):
                prefix_no_save.append(msg)
            else:
                break

        # 6. 执行切片：保留预设前缀 + 最近 L 条历史 User 消息及其附属的全部 Assistant、Tool 消息
        # 裁剪后的普通消息随 AstrBot 正常流程写回数据库，_no_save 消息框架自动跳过不入库
        evicted_turns = total_user_turns - L
        contexts[:] = prefix_no_save + contexts[cut_idx:]
        self._stats["eviction_count"] += 1
        self._stats["evicted_user_turns"] += evicted_turns

        if self._debug_log_enabled():
            logger.debug(
                f"[CacheOptimizer] 触发阶梯式裁剪: 历史 User 轮数 {total_user_turns} >= {L + S} -> 裁掉前 {evicted_turns} 轮 -> 保留最近 {L} 轮历史 (消息数 {len(contexts)})"
            )

    # ------------------------------------------------------------------
    # 真实 Token Usage 统计监听 (DeepSeek 原生返回)
    # ------------------------------------------------------------------
    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp: Any) -> None:
        if not self._is_enabled() or not resp:
            return

        usage = getattr(resp, "usage", None)
        if not usage:
            return

        cached = getattr(usage, "input_cached", 0)
        other = getattr(usage, "input_other", 0)
        cached = int(cached) if isinstance(cached, (int, float)) else 0
        other = int(other) if isinstance(other, (int, float)) else 0

        if cached > 0 or other > 0:
            self._stats["total_cached_tokens"] += cached
            self._stats["total_other_tokens"] += other
            self._stats["response_count"] += 1

    # ------------------------------------------------------------------
    # 状态与诊断指令（双重管理员权限保护）
    # ------------------------------------------------------------------
    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("cache_opt")
    async def cmd_cache_opt(self, event: AstrMessageEvent) -> None:
        """查看提示词缓存保活与真实 Token 统计（仅管理员）"""
        if hasattr(event, "is_admin") and callable(event.is_admin) and not event.is_admin():
            yield event.plain_result("⚠️ 只有管理员可执行此指令。")
            return

        total_reqs = self._stats["total_requests"]
        eviction_count = self._stats["eviction_count"]
        evicted_user_turns = self._stats["evicted_user_turns"]
        cached_tokens = self._stats["total_cached_tokens"]
        other_tokens = self._stats["total_other_tokens"]
        total_tokens = cached_tokens + other_tokens

        if total_tokens > 0:
            hit_rate_str = f"{(cached_tokens / total_tokens * 100):.2f}%"
        else:
            hit_rate_str = "暂无数据"

        threshold = self._low_watermark_turns() + self._step_turns()

        # 实时检测当前会话的框架配置健康状态
        status, diag_msgs = self._diagnose_framework_config(getattr(event, "unified_msg_origin", None))
        if status == "warning":
            diag_str = "【配置健康诊断】⚠️ 检测到配置风险：\n" + "\n".join(f"- {w}" for w in diag_msgs)
        elif status == "healthy":
            diag_str = "【配置健康诊断】🟢 当前会话框架配置无冲突"
        else:
            diag_str = f"【配置健康诊断】⚪ {' '.join(diag_msgs)}"

        msg = (
            f"=== 提示词缓存保活运行状态 ===\n"
            f"运行状态: {'🟢 正常生效中' if self._is_enabled() else '🔴 已停用'}\n"
            f"保留轮数 (L): {self._low_watermark_turns()} 轮\n"
            f"清理步长 (S): {self._step_turns()} 轮 (历史满 {threshold} 轮触发裁剪)\n"
            f"历史裁剪触发: {eviction_count} 次 (累计淘汰 {evicted_user_turns} 轮旧历史)\n"
            f"全局累计处理 LLM 请求: {total_reqs} 次 (不代表单会话轮数)\n"
            f"已检查会话数: {len(self._checked_umos)} 个\n"
            f"---------------------------------\n"
            f"{diag_str}\n"
            f"---------------------------------\n"
            f"【最终 LLM 请求缓存统计 (DeepSeek Usage)】\n"
            f"命中缓存 Tokens: {cached_tokens:,}\n"
            f"未命中 Tokens: {other_tokens:,}\n"
            f"总计输入 Tokens: {total_tokens:,}\n"
            f"加权平均命中率: {hit_rate_str}\n"
            f"注: 该统计来源于模型服务端返回的真实 Usage，反映每次发车的最终请求缓存命中情况。\n"
            f"================================="
        )
        yield event.plain_result(msg)
