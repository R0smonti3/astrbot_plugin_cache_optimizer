from __future__ import annotations

import copy
import re
import time
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star

try:
    from astrbot.core.agent.message import TextPart
except ImportError:
    TextPart = None


def _safe_mark_temp(part: Any) -> Any:
    """安全调用 mark_as_temp 或注入 _no_save 属性，防止旧版 AstrBot 缺失方法崩溃。"""
    if hasattr(part, "mark_as_temp") and callable(getattr(part, "mark_as_temp")):
        try:
            return part.mark_as_temp()
        except Exception:
            pass
    if hasattr(part, "_no_save"):
        try:
            setattr(part, "_no_save", True)
        except Exception:
            pass
    return part


class CacheOptimizerPlugin(Star):
    """上下文缓存保活插件：
    1. 分块修剪历史：积攒一批再一次性清理老消息，保持前序上下文起点固定，减少缓存穿透；
    2. 严格对齐用户发言：确保每次修剪后的第一条消息为合法的用户发言，防止接口报 400 错误；
    3. 固定系统提示词：把动态变化的当前时间移到当轮用户提问末尾，避免开头内容频繁变动击穿缓存；
    4. 自动放宽框架上限：运行时自检原生会话配置，防止框架提前截断。
    """

    def __init__(self, context: Context, config: dict | None = None) -> None:
        super().__init__(context)
        self.config = config if config is not None else {}
        self._patched_umos: set[str] = set()
        self._last_anchor_sig: dict[str, str] = {}
        self._stats = {
            "total_requests": 0,
            "system_frozen_count": 0,
            "chunk_evicted_count": 0,
            "anchor_hits": 0,
        }

        # 严格限制带明确前缀的时间戳与动态标签，绝不使用裸日期正则以防误伤人设背景时间
        self._time_regexes = [
            re.compile(
                r"(?:当前时间|现在时间|北京时间|系统时间|当前日期|Current\s+(?:date\s+and\s+time|datetime|time|date))[：:]\s*[^\n]+",
                re.IGNORECASE,
            ),
            re.compile(r"<system_reminder>.*?</system_reminder>", re.DOTALL),
        ]

    # ------------------------------------------------------------------
    # 配置读取辅助
    # ------------------------------------------------------------------
    def _is_enabled(self) -> bool:
        return bool(self.config.get("enabled", True))

    def _low_watermark_turns(self) -> int:
        return max(4, int(self.config.get("low_watermark_turns", 18)))

    def _step_turns(self) -> int:
        return max(2, int(self.config.get("step_turns", 12)))

    def _freeze_system_prompt_enabled(self) -> bool:
        return bool(self.config.get("freeze_system_prompt", True))

    def _auto_patch_enabled(self) -> bool:
        return bool(self.config.get("auto_patch_framework_config", True))

    def _debug_log_enabled(self) -> bool:
        return bool(self.config.get("debug_log", False))

    # ------------------------------------------------------------------
    # 框架原生配置自检与热修复（按 umo 记录，避免多会话漏检）
    # ------------------------------------------------------------------
    def _check_and_patch_framework_config(self, umo: str | None = None) -> None:
        umo_key = str(umo or "global")
        if umo_key in self._patched_umos:
            return
        self._patched_umos.add(umo_key)

        try:
            cfg = self.context.get_config(umo=umo)
            if not cfg:
                return

            agent_runner = cfg.get("agent_runner")
            if not isinstance(agent_runner, dict):
                return
            runner_config = agent_runner.get("config")
            if not isinstance(runner_config, dict):
                return
            compression = runner_config.get("compression")
            if not isinstance(compression, dict):
                return

            max_turns = compression.get("max_turns", -1)
            high_watermark = self._low_watermark_turns() + self._step_turns()

            if 0 < max_turns <= high_watermark:
                if self._auto_patch_enabled():
                    suggested_max = high_watermark + 40
                    compression["max_turns"] = suggested_max
                    logger.warning(
                        f"[CacheOptimizer] [警告] 检测到 AstrBot 原生 max_turns={max_turns} 小于等于插件高水位({high_watermark}轮)！"
                        f"已自动在会话 [{umo_key}] 内存中调大至 {suggested_max} 轮，防止原生滑动窗口提前 popleft 击穿 Prefix Cache。"
                    )
                else:
                    logger.warning(
                        f"[CacheOptimizer] [警告] AstrBot 原生 max_turns={max_turns} 小于等于插件高水位({high_watermark}轮)！"
                        f"原生滑动窗口将在插件触发前抢先淘汰历史，导致分块缓存失效。建议在 WebUI 将上下文轮数上限调大或开启自动修复。"
                    )

            strategy = compression.get("overflow_strategy", "")
            if strategy == "llm_compress":
                logger.warning(
                    "[CacheOptimizer] [警告] 检测到 AstrBot 原生会话溢出策略为 'llm_compress' (LLM摘要压缩)！"
                    "该策略在满额时会调用额外模型重写历史，彻底粉碎后续所有请求的 Prefix Cache。"
                    "强烈建议在 WebUI 的会话设置中将溢出策略调整为 'truncate_by_turns'。"
                )
        except Exception as exc:
            logger.debug(f"[CacheOptimizer] 框架配置检测跳过: {exc}")

    # ------------------------------------------------------------------
    # 核心拦截入口：发车前的最后一道闸门 (priority=-9999)
    # ------------------------------------------------------------------
    @filter.on_llm_request(priority=-9999)
    async def on_llm_request(self, event: AstrMessageEvent, req: Any) -> None:
        if not self._is_enabled() or not req:
            return

        self._stats["total_requests"] += 1
        umo = event.unified_msg_origin
        self._check_and_patch_framework_config(umo)

        # 步骤 1：System Prompt 冰封与动态信息剥离下沉
        if self._freeze_system_prompt_enabled():
            self._apply_system_prompt_freezer(req)

        # 步骤 2：历史上下文步长取模锚定与 User 角色对齐切片
        self._apply_chunked_eviction(req, umo)

    # ------------------------------------------------------------------
    # 步骤 1：System Prompt 冰封器（剥离时间戳并阅后即焚下沉）
    # ------------------------------------------------------------------
    def _apply_system_prompt_freezer(self, req: Any) -> None:
        sys_prompt = getattr(req, "system_prompt", None)
        if not sys_prompt or not isinstance(sys_prompt, str):
            return

        stripped_parts: list[str] = []
        cleaned_prompt = sys_prompt

        for pattern in self._time_regexes:
            matches = pattern.findall(cleaned_prompt)
            if matches:
                for m in matches:
                    text_match = m.strip() if isinstance(m, str) else str(m).strip()
                    if text_match:
                        stripped_parts.append(text_match)
                cleaned_prompt = pattern.sub("", cleaned_prompt)

        if not stripped_parts:
            return

        # 规范化清理后的静态 System Prompt（剔除连续空行，保持 100% 恒定常量）
        cleaned_prompt = re.sub(r"\n{3,}", "\n\n", cleaned_prompt).strip()
        req.system_prompt = cleaned_prompt
        self._stats["system_frozen_count"] += 1

        # 将剥离出的动态时钟与环境信息下沉至当轮用户末尾（使用 mark_as_temp 绝不存库）
        dynamic_block = (
            "[时钟与环境动态感知（已从系统提示词隔离下沉）]：\n"
            + "\n".join(stripped_parts)
        )

        if (
            TextPart is not None
            and hasattr(req, "extra_user_content_parts")
            and isinstance(req.extra_user_content_parts, list)
        ):
            part = TextPart(text=f"\n\n{dynamic_block}")
            req.extra_user_content_parts.append(_safe_mark_temp(part))

        if self._debug_log_enabled():
            logger.debug(
                f"[CacheOptimizer] System Prompt 动态信息已剥离并下沉 ({len(stripped_parts)} 处)，静态长度: {len(cleaned_prompt)}"
            )

    # ------------------------------------------------------------------
    # 步骤 2：步长取模锚定切片与 User 角色边界对齐（核心算法）
    # ------------------------------------------------------------------
    def _apply_chunked_eviction(self, req: Any, umo: str) -> None:
        contexts = getattr(req, "contexts", None)
        if not contexts or not isinstance(contexts, list):
            return

        total_msgs = len(contexts)
        low_watermark_turns = self._low_watermark_turns()
        step_turns = self._step_turns()

        # 1 轮对应 2 条消息 (user + assistant)
        L = low_watermark_turns * 2
        S = step_turns * 2

        # 历史条数还在低水位+步长内，无需任何裁剪，全量保留
        if total_msgs <= L + S:
            start_idx = 0
        else:
            # 【核心数学公式】：精准步长取模锚定（将余数映射为 1~S 的步进保留）
            # 在 total_msgs 从 L+S+1 增长到 L+2S 期间，raw_start_idx 保持绝对恒定！
            keep = L + ((total_msgs - L - 1) % S) + 1
            raw_start_idx = total_msgs - keep

            if raw_start_idx <= 0:
                start_idx = 0
            else:
                # 【防 400 报错对齐红线】：强制向后扫描，对齐至合法的普通 user 消息起点
                start_idx = raw_start_idx
                while start_idx < total_msgs:
                    msg = contexts[start_idx]
                    if not isinstance(msg, dict):
                        start_idx += 1
                        continue
                    role = msg.get("role")
                    has_tool_call_id = bool(msg.get("tool_call_id"))
                    # 必须是 role == "user" 且不能是 tool_call_id 孤儿返回
                    if role == "user" and not has_tool_call_id:
                        break
                    start_idx += 1

                # 极端防穿透保底：若向后扫描过头导致剩余消息不足 2 条，向前安全回退找最近的 user 消息
                if total_msgs - start_idx < 2 and total_msgs >= 2:
                    fallback_found = False
                    for fb_idx in range(max(0, raw_start_idx - 4), total_msgs):
                        m = contexts[fb_idx]
                        if (
                            isinstance(m, dict)
                            and m.get("role") == "user"
                            and not m.get("tool_call_id")
                        ):
                            start_idx = fb_idx
                            fallback_found = True
                            break
                    if not fallback_found:
                        start_idx = 0

        # 执行内存原地切片赋值（确保引用与下游读取完全同步）
        if start_idx > 0:
            contexts[:] = contexts[start_idx:]

        # 【真实前缀锚点命中判定】：对比当前保留的第一条消息特征签名
        # 只要切片后物理第一条消息未变（哪怕 start_idx > 0），前缀缓存就是 100% 命中的！
        curr_first = contexts[0] if contexts else {}
        curr_sig = (
            f"{curr_first.get('role')}:{str(curr_first.get('content'))[:64]}"
            if isinstance(curr_first, dict)
            else ""
        )
        prev_sig = self._last_anchor_sig.get(umo)
        self._last_anchor_sig[umo] = curr_sig

        if prev_sig is None:
            # 首次记录锚点，计为前缀稳定基准
            self._stats["anchor_hits"] += 1
        elif curr_sig == prev_sig:
            # 首条消息签名与上一轮完全一致：说明前缀锚点未动，100% 缓存命中！
            self._stats["anchor_hits"] += 1
            if self._debug_log_enabled():
                logger.debug(
                    f"[CacheOptimizer] 会话 [{umo}] 锚点签名未位移 ({curr_sig[:24]}...)，前缀缓存 100% 命中"
                )
        else:
            # 首条消息签名发生改变：说明切点发生了跨步长跳跃淘汰（缓存发生重置）
            self._stats["chunk_evicted_count"] += 1
            if self._debug_log_enabled():
                logger.debug(
                    f"[CacheOptimizer] 会话 [{umo}] 触发分块跳跃淘汰: 签名更新为 {curr_sig[:24]}...，保留 {len(contexts)} 条"
                )

    # ------------------------------------------------------------------
    # 状态与诊断指令（双重管理员权限保护）
    # ------------------------------------------------------------------
    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("cache_opt")
    async def cmd_cache_opt(self, event: AstrMessageEvent) -> None:
        """查看 Prompt Cache 优化器运行指标与状态（仅管理员）"""
        if hasattr(event, "is_admin") and callable(event.is_admin) and not event.is_admin():
            yield event.plain_result("⚠️ 只有管理员可执行此指令。")
            return

        total = self._stats["total_requests"]
        frozen = self._stats["system_frozen_count"]
        evicted = self._stats["chunk_evicted_count"]
        hits = self._stats["anchor_hits"]

        msg = (
            f"=== 提高缓存命中运行状态 ===\n"
            f"运行状态: {'🟢 正常生效中' if self._is_enabled() else '🔴 已停用'}\n"
            f"清理保留轮数: {self._low_watermark_turns()} 轮\n"
            f"分块清理步长: {self._step_turns()} 轮\n"
            f"系统提示词固定: {'已开启 (动态时间移至末尾)' if self._freeze_system_prompt_enabled() else '已关闭'}\n"
            f"累计优化请求: {total} 次\n"
            f"时间下沉次数: {frozen} 次\n"
            f"前缀未位移轮次: {hits} 次\n"
            f"历史修剪跃迁: {evicted} 次\n"
            f"已接管会话数: {len(self._patched_umos)} 个\n"
            f"========================="
        )
        yield event.plain_result(msg)
