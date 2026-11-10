from __future__ import annotations

import copy
import re
import time
from datetime import datetime
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


def _extract_usage_tokens(usage: Any) -> tuple[int, int]:
    """统一提取多模型厂商的 (cached_tokens, uncached_tokens)。
    支持 DeepSeek, OpenAI, Claude, Gemini 以及字典或对象。
    """
    if not usage:
        return 0, 0
    get = lambda k: usage.get(k, 0) if isinstance(usage, dict) else getattr(usage, k, 0)

    # 1. DeepSeek 规范 (input_cached, input_other)
    c = get("input_cached")
    o = get("input_other")
    if c or o:
        return int(c or 0), int(o or 0)

    # 2. OpenAI 规范 (prompt_tokens_details.cached_tokens)
    pt = get("prompt_tokens") or get("prompt_token_count") or get("input_tokens") or 0
    details = get("prompt_tokens_details")
    if details:
        det_get = lambda k: details.get(k, 0) if isinstance(details, dict) else getattr(details, k, 0)
        c = det_get("cached_tokens")

    # 3. Claude / Gemini 规范
    if not c:
        c = get("cache_read_input_tokens") or get("cached_content_token_count")

    cached = int(c or 0)
    total_prompt = int(pt or 0)
    uncached = max(0, total_prompt - cached) if total_prompt >= cached else 0
    return cached, uncached


_TIMESTAMP_PATTERN = re.compile(
    r"(?:当前时间|北京时间|系统时间|时间戳|current\s*time)[:：\s]*\d{4}[-/年]\d{1,2}[-/月]\d{1,2}.*?\d{1,2}[:：]\d{1,2}",
    re.IGNORECASE,
)


def _check_system_prompt_timestamp(system_prompt: str | None) -> list[str]:
    """检测系统提示词中是否存在秒/分级高频动态时间戳（破坏前缀缓存）。"""
    if not system_prompt or not isinstance(system_prompt, str):
        return []
    if _TIMESTAMP_PATTERN.search(system_prompt):
        return [
            "系统提示词 (System Prompt) 中检测到秒/分级动态时间戳。"
            "根据 KV Cache 前缀缓存原理，提示词前缀只要变动 1 个字符，其后的所有对话上下文缓存将全量击穿失效。"
            "强烈建议将动态时间戳移至最新用户消息末尾，而不是写在 System Prompt 中。"
        ]
    return []


def _format_eviction_time(ts: float | int) -> str:
    """将剪切时间戳格式化为可读字符串，未发生过则返回 '暂无剪切记录'。"""
    if not ts or ts <= 0:
        return "暂无剪切记录"
    diff = max(0, int(time.time() - ts))
    if diff < 60:
        ago = f"{diff} 秒前"
    elif diff < 3600:
        ago = f"{diff // 60} 分 {diff % 60} 秒前"
    elif diff < 86400:
        ago = f"{diff // 3600} 小时 {(diff % 3600) // 60} 分前"
    else:
        ago = f"{diff // 86400} 天前"
    dt_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    return f"{dt_str} ({ago})"


class CacheOptimizerPlugin(Star):
    """大模型提示词缓存保活插件 (CacheOptimizer)：
    1. 阶梯式历史裁剪：按合法的普通 User 轮数精准计数，当历史累计满 L + S 轮时触发裁剪，
       一次性修剪至保留最近 L 轮历史，裁剪结果通过 AstrBot 正常 Agent 保存流程持久化写回数据库；
    2. 严格事务边界对齐：裁剪起点锁定在完整 User 消息边界，完整保留其后附属的 Assistant 与 Tool 调用链路，
       杜绝工具调用事务被截断导致的 400 报错；
    3. 多厂商 Token Usage 监控：兼容 DeepSeek、OpenAI、Claude、Gemini 的 Prompt Cache 命中率统计；
    4. 高频动态时间戳与原生配置冲突诊断：实时检测秒级时间戳击穿前缀与原生 max_turns 截断风险；
    5. 支持全局与分会话独立查看运行指标 (/cache_opt [here]) 及 Web 可视化看板。
    """

    def __init__(self, context: Context, config: dict | None = None) -> None:
        super().__init__(context)
        self.config = config if config is not None else {}
        self._checked_umos: set[str] = set()
        self._stats: dict[str, Any] = {
            "total_requests": 0,
            "eviction_count": 0,
            "evicted_user_turns": 0,
            "total_cached_tokens": 0,
            "total_other_tokens": 0,
            "response_count": 0,
            "last_eviction_ts": 0.0,
            "last_eviction_turns": 0,
            "last_eviction_umo": "",
        }
        self._session_stats: dict[str, dict[str, Any]] = {}
        self._eviction_logs: list[dict[str, Any]] = []

    async def initialize(self) -> None:
        for route in ("/astrbot_plugin_cache_optimizer/status", "/cache_optimizer/status"):
            try:
                self.context.register_web_api(
                    route, self._api_status, ["GET"], "缓存优化状态"
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[CacheOptimizer] 注册状态接口失败: {exc}")

    def _get_session_stat(self, umo: str) -> dict[str, Any]:
        if umo not in self._session_stats:
            if len(self._session_stats) >= 200:
                self._session_stats.pop(next(iter(self._session_stats)))
            self._session_stats[umo] = {
                "requests": 0,
                "evictions": 0,
                "evicted_turns": 0,
                "cached_tokens": 0,
                "other_tokens": 0,
                "current_user_turns": 0,
                "last_request_ts": 0.0,
                "last_eviction_ts": 0.0,
                "last_eviction_turns": 0,
            }
        return self._session_stats[umo]

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
    # 框架原生配置与前缀安全诊断（支持实时检查与会话按需查询）
    # ------------------------------------------------------------------
    def _diagnose_framework_config(
        self, umo: str | None = None, system_prompt: str | None = None
    ) -> tuple[str, list[str]]:
        """实时检查指定会话的 AstrBot 框架配置冲突并返回 (状态, 描述列表)。"""
        warnings: list[str] = []
        if system_prompt:
            warnings.extend(_check_system_prompt_timestamp(system_prompt))

        try:
            cfg = self.context.get_config(umo=umo)
            if cfg and isinstance(cfg, dict):
                agent_runner = cfg.get("agent_runner")
                if isinstance(agent_runner, dict):
                    runner_config = agent_runner.get("config")
                    if isinstance(runner_config, dict):
                        compression = runner_config.get("compression")
                        if isinstance(compression, dict):
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
        except Exception as exc:
            logger.debug(f"[CacheOptimizer] 框架配置检测跳过: {exc}")

        if warnings:
            return "warning", warnings
        return "healthy", []

    def _check_framework_config(self, umo: str | None = None, system_prompt: str | None = None) -> None:
        umo_key = str(umo or "global")
        if umo_key in self._checked_umos:
            return
        self._checked_umos.add(umo_key)

        status, warnings = self._diagnose_framework_config(umo, system_prompt)
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

        now_ts = time.time()
        self._stats["total_requests"] += 1
        umo = str(getattr(event, "unified_msg_origin", "") or "default")
        s_stat = self._get_session_stat(umo)
        s_stat["requests"] += 1
        s_stat["last_request_ts"] = now_ts
        self._check_framework_config(umo, getattr(req, "system_prompt", None))

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

        if umo:
            self._get_session_stat(umo)["current_user_turns"] = total_user_turns

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
        now_ts = time.time()
        evicted_turns = total_user_turns - L
        contexts[:] = prefix_no_save + contexts[cut_idx:]

        self._stats["eviction_count"] += 1
        self._stats["evicted_user_turns"] += evicted_turns
        self._stats["last_eviction_ts"] = now_ts
        self._stats["last_eviction_turns"] = evicted_turns
        self._stats["last_eviction_umo"] = umo

        if umo:
            s_stat = self._get_session_stat(umo)
            s_stat["evictions"] += 1
            s_stat["evicted_turns"] += evicted_turns
            s_stat["current_user_turns"] = L
            s_stat["last_eviction_ts"] = now_ts
            s_stat["last_eviction_turns"] = evicted_turns

        self._eviction_logs.append(
            {
                "ts": now_ts,
                "umo": umo or "default",
                "before_turns": total_user_turns,
                "after_turns": L,
                "evicted_turns": evicted_turns,
            }
        )
        if len(self._eviction_logs) > 20:
            del self._eviction_logs[: len(self._eviction_logs) - 20]

        if self._debug_log_enabled():
            logger.debug(
                f"[CacheOptimizer] 触发阶梯式裁剪: 历史 User 轮数 {total_user_turns} >= {L + S} -> 裁掉前 {evicted_turns} 轮 -> 保留最近 {L} 轮历史 (消息数 {len(contexts)})"
            )

    # ------------------------------------------------------------------
    # 真实 Token Usage 统计监听 (兼容 DeepSeek, OpenAI, Claude, Gemini)
    # ------------------------------------------------------------------
    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp: Any) -> None:
        if not self._is_enabled() or not resp:
            return

        usage = getattr(resp, "usage", None)
        cached, other = _extract_usage_tokens(usage)

        if cached > 0 or other > 0:
            self._stats["total_cached_tokens"] += cached
            self._stats["total_other_tokens"] += other
            self._stats["response_count"] += 1

            umo = str(getattr(event, "unified_msg_origin", "") or "default")
            s_stat = self._get_session_stat(umo)
            s_stat["cached_tokens"] += cached
            s_stat["other_tokens"] += other

    # ------------------------------------------------------------------
    # Web 可视化看板状态接口
    # ------------------------------------------------------------------
    def _status_payload(self) -> dict[str, Any]:
        now_ts = time.time()
        L = self._low_watermark_turns()
        S = self._step_turns()
        threshold = L + S

        cached_tokens = int(self._stats["total_cached_tokens"])
        other_tokens = int(self._stats["total_other_tokens"])
        total_tokens = cached_tokens + other_tokens
        hit_rate_pct = round(cached_tokens / total_tokens * 100, 2) if total_tokens > 0 else None

        last_ev_ts = float(self._stats.get("last_eviction_ts") or 0.0)
        diag_status, diag_msgs = self._diagnose_framework_config(None)

        sessions = []
        for umo, s in self._session_stats.items():
            s_cached = int(s["cached_tokens"])
            s_other = int(s["other_tokens"])
            s_total = s_cached + s_other
            s_ev_ts = float(s.get("last_eviction_ts") or 0.0)
            s_req_ts = float(s.get("last_request_ts") or 0.0)
            sessions.append(
                {
                    "umo": umo,
                    "requests": s["requests"],
                    "evictions": s["evictions"],
                    "evicted_turns": s["evicted_turns"],
                    "current_user_turns": s.get("current_user_turns", 0),
                    "cached_tokens": s_cached,
                    "other_tokens": s_other,
                    "total_tokens": s_total,
                    "hit_rate_pct": round(s_cached / s_total * 100, 2) if s_total > 0 else None,
                    "last_request_ago": max(0, int(now_ts - s_req_ts)) if s_req_ts > 0 else None,
                    "last_eviction_ts": s_ev_ts,
                    "last_eviction_ago": max(0, int(now_ts - s_ev_ts)) if s_ev_ts > 0 else None,
                    "last_eviction_time_str": _format_eviction_time(s_ev_ts),
                    "last_eviction_turns": s.get("last_eviction_turns", 0),
                }
            )
        sessions.sort(key=lambda x: x["last_request_ago"] if x["last_request_ago"] is not None else 9999999)

        logs = [
            {
                "ts": item["ts"],
                "time_str": datetime.fromtimestamp(item["ts"]).strftime("%Y-%m-%d %H:%M:%S"),
                "ago": max(0, int(now_ts - item["ts"])),
                "umo": item["umo"],
                "before_turns": item["before_turns"],
                "after_turns": item["after_turns"],
                "evicted_turns": item["evicted_turns"],
            }
            for item in reversed(self._eviction_logs)
        ]

        return {
            "enabled": self._is_enabled(),
            "low_watermark_turns": L,
            "step_turns": S,
            "threshold_turns": threshold,
            "total_requests": self._stats["total_requests"],
            "eviction_count": self._stats["eviction_count"],
            "evicted_user_turns": self._stats["evicted_user_turns"],
            "response_count": self._stats["response_count"],
            "total_cached_tokens": cached_tokens,
            "total_other_tokens": other_tokens,
            "total_tokens": total_tokens,
            "hit_rate_pct": hit_rate_pct,
            "last_eviction_ts": last_ev_ts,
            "last_eviction_ago": max(0, int(now_ts - last_ev_ts)) if last_ev_ts > 0 else None,
            "last_eviction_time_str": _format_eviction_time(last_ev_ts),
            "last_eviction_turns": self._stats.get("last_eviction_turns", 0),
            "last_eviction_umo": self._stats.get("last_eviction_umo", ""),
            "diag_status": diag_status,
            "diag_messages": diag_msgs,
            "sessions": sessions,
            "eviction_logs": logs,
        }

    async def _api_status(self):
        return self._status_payload()

    # ------------------------------------------------------------------
    # 状态与诊断指令（双重管理员权限保护，支持 /cache_opt here）
    # ------------------------------------------------------------------
    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("cache_opt")
    async def cmd_cache_opt(self, event: AstrMessageEvent, target_scope: str = "") -> None:
        """查看提示词缓存保活与真实 Token 统计（仅管理员，支持 /cache_opt here）"""
        if hasattr(event, "is_admin") and callable(event.is_admin) and not event.is_admin():
            yield event.plain_result("⚠️ 只有管理员可执行此指令。")
            return

        scope = target_scope.strip().lower() if target_scope else ""
        if not scope:
            msg_str = getattr(event, "message_str", "") or ""
            if any(w in msg_str.lower() for w in ("here", "当前", "本群")):
                scope = "here"

        umo = str(getattr(event, "unified_msg_origin", "") or "")
        is_here = (scope in ("here", "当前", "本群")) and bool(umo)

        if is_here:
            s_stat = self._get_session_stat(umo)
            reqs = s_stat["requests"]
            evictions = s_stat["evictions"]
            evicted_turns = s_stat["evicted_turns"]
            cached_tokens = s_stat["cached_tokens"]
            other_tokens = s_stat["other_tokens"]
            last_ev_ts = float(s_stat.get("last_eviction_ts") or 0.0)
            last_ev_str = _format_eviction_time(last_ev_ts)
            if last_ev_ts > 0 and s_stat.get("last_eviction_turns"):
                last_ev_str += f" (裁掉 {s_stat['last_eviction_turns']} 轮)"
            title_prefix = f"【当前会话独立缓存统计】({umo})"
        else:
            reqs = self._stats["total_requests"]
            evictions = self._stats["eviction_count"]
            evicted_turns = self._stats["evicted_user_turns"]
            cached_tokens = self._stats["total_cached_tokens"]
            other_tokens = self._stats["total_other_tokens"]
            last_ev_ts = float(self._stats.get("last_eviction_ts") or 0.0)
            last_ev_str = _format_eviction_time(last_ev_ts)
            if last_ev_ts > 0 and self._stats.get("last_eviction_turns"):
                ev_umo = self._stats.get("last_eviction_umo") or ""
                umo_note = f", 会话 {ev_umo}" if ev_umo else ""
                last_ev_str += f" (裁掉 {self._stats['last_eviction_turns']} 轮{umo_note})"
            title_prefix = f"【全局累计缓存保活统计】(活跃会话数: {len(self._session_stats)})"

        total_tokens = cached_tokens + other_tokens
        hit_rate_str = f"{(cached_tokens / total_tokens * 100):.2f}%" if total_tokens > 0 else "暂无数据"
        threshold = self._low_watermark_turns() + self._step_turns()

        # 实时检测当前会话的框架配置健康状态
        status, diag_msgs = self._diagnose_framework_config(umo)
        if status == "warning":
            diag_str = "【配置健康诊断】⚠️ 检测到配置风险：\n" + "\n".join(f"- {w}" for w in diag_msgs)
        elif status == "healthy":
            diag_str = "【配置健康诊断】🟢 当前会话框架配置无冲突"
        else:
            diag_str = f"【配置健康诊断】⚪ {' '.join(diag_msgs)}"

        scope_tip = (
            "💡 提示：输入 `/cache_opt` 可查看全局总统计"
            if is_here
            else "💡 提示：输入 `/cache_opt here` 可查看当前会话的独立缓存统计"
        )

        msg = (
            f"=== 提示词缓存保活运行状态 ===\n"
            f"运行状态: {'🟢 正常生效中' if self._is_enabled() else '🔴 已停用'}\n"
            f"保留轮数 (L): {self._low_watermark_turns()} 轮 | 清理步长 (S): {self._step_turns()} 轮 (历史满 {threshold} 轮裁剪)\n"
            f"---------------------------------\n"
            f"{diag_str}\n"
            f"---------------------------------\n"
            f"{title_prefix}\n"
            f"处理 LLM 请求: {reqs} 次 | 历史裁剪触发: {evictions} 次 (累计淘汰 {evicted_turns} 轮旧历史)\n"
            f"上一轮剪切时间: {last_ev_str}\n"
            f"命中缓存 Tokens: {cached_tokens:,}\n"
            f"未命中 Tokens: {other_tokens:,}\n"
            f"总计输入 Tokens: {total_tokens:,}\n"
            f"加权平均命中率: {hit_rate_str}\n"
            f"---------------------------------\n"
            f"{scope_tip}\n"
            f"================================="
        )
        yield event.plain_result(msg)
