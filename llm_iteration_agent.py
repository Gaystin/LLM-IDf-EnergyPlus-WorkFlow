"""LLM迭代效率智能体：基于历史迭代结果生成下一轮策略提示。"""

from typing import Dict, List


class LLMIterationEfficiencyAgent:
    """根据历史迭代表现，构建“尽快达标”的策略提示。"""

    def __init__(self, target_saving_pct: float = 50.0):
        self.target_saving_pct = float(target_saving_pct)

    @staticmethod
    def _safe_saving_pct(baseline: float, current: float) -> float:
        try:
            b = float(baseline)
            c = float(current)
        except Exception:
            return 0.0
        if b <= 0:
            return 0.0
        return (b - c) / b * 100.0

    def build_directive(
        self,
        iteration_history: List[Dict],
        field_modification_history: Dict[str, int],
        field_effectiveness: Dict[str, Dict] = None,
    ) -> str:
        if not iteration_history:
            return (
                "【效率智能体建议】首轮不预设固定优化方向。"
                " 请基于当前城市与模型的候选对象字段，自主评估各方向的预期节能收益。"
            )

        baseline_metrics = iteration_history[0].get("metrics", {}) if iteration_history else {}
        baseline_total = float(baseline_metrics.get("total_site_energy_kwh", 0.0) or 0.0)

        saving_series = []
        for item in iteration_history:
            m = item.get("metrics", {})
            total = float(m.get("total_site_energy_kwh", 0.0) or 0.0)
            saving_series.append(self._safe_saving_pct(baseline_total, total))

        best_saving = max(saving_series) if saving_series else 0.0
        latest_saving = saving_series[-1] if saving_series else 0.0
        remaining = max(0.0, self.target_saving_pct - latest_saving)

        lines = [
            "【效率智能体建议】",
            f"- 当前最佳节能率: {best_saving:.2f}%",
            f"- 当前最新节能率: {latest_saving:.2f}%",
            f"- 与目标{self.target_saving_pct:.2f}%的差距: {remaining:.2f}%",
            "- 目标是以更少迭代轮次达标，优先给出高收益且可执行的字段修改组合。",
            "- 每轮都应推动总节能率向50%单调逼近，避免在阈值附近来回震荡。",
        ]

        if 0.0 < remaining <= 5.0:
            lines.append("- 当前已接近目标阈值：请进入冲刺模式，减少低收益探索，优先跨越50%目标。")

        if len(saving_series) >= 2:
            delta = saving_series[-1] - saving_series[-2]
            lines.append(f"- 最近一轮节能率增益: {delta:+.2f}个百分点")
            if delta <= 0.5:
                lines.append("- 最近增益偏小：建议减少低收益尝试，优先选择预期效果更强的方向。")
            if delta < 0:
                lines.append("- 最近一轮出现反优化：本轮必须显式纠偏，优先替换为历史平均收益更高的字段组合。")

        if field_modification_history:
            top_fields = sorted(
                field_modification_history.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:5]
            if top_fields:
                lines.append("- 历史高频字段（仅供参考，不做重复率硬限制）：")
                for field_key, count in top_fields:
                    lines.append(f"  * {field_key}: {int(count)}次")

        if isinstance(field_effectiveness, dict) and field_effectiveness:
            ranked = []
            for key, rec in field_effectiveness.items():
                try:
                    uses = int(rec.get('uses', 0) or 0)
                    total_delta = float(rec.get('total_delta_kwh', 0.0) or 0.0)
                except Exception:
                    continue
                if uses <= 0:
                    continue
                avg_delta = total_delta / uses
                ranked.append((key, avg_delta, total_delta, uses))

            if ranked:
                top_good = sorted(ranked, key=lambda x: x[1], reverse=True)[:3]
                top_bad = sorted(ranked, key=lambda x: x[1])[:2]

                lines.append("- 历史平均收益较高字段（优先参考，不是硬限制）：")
                for key, avg_delta, total_delta, uses in top_good:
                    lines.append(
                        f"  * {key}: 平均{avg_delta:+.2f} kWh/次，累计{total_delta:+.2f} kWh（{uses}次）"
                    )

                lines.append("- 历史平均收益较差字段（谨慎评估，不是硬限制）：")
                for key, avg_delta, total_delta, uses in top_bad:
                    lines.append(
                        f"  * {key}: 平均{avg_delta:+.2f} kWh/次，累计{total_delta:+.2f} kWh（{uses}次）"
                    )

        lines.append("- 为尽快达到50%目标，可适度提高有效方向的推进力度，但避免过于激进的步长。")
        return "\n".join(lines)
