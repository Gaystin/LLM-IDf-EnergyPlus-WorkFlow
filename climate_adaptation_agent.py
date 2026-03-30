"""气候适应智能体：从EPW提取城市气候信息，供LLM自主判断优化侧重。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ClimateSnapshot:
    city_name: str
    latitude: Optional[float]
    longitude: Optional[float]
    elevation_m: Optional[float]
    annual_mean_db_c: Optional[float]
    annual_mean_rh_pct: Optional[float]
    annual_hdd18_c_day: Optional[float]
    annual_cdd26_c_day: Optional[float]
    winter_mean_db_c: Optional[float]
    summer_mean_db_c: Optional[float]


class ClimateAdaptationAgent:
    """解析EPW并输出给LLM的气候上下文，不内置城市-字段硬编码策略。"""

    def __init__(self) -> None:
        self._cache: Dict[str, ClimateSnapshot] = {}

    @staticmethod
    def _safe_float(value: str) -> Optional[float]:
        try:
            return float(str(value).strip())
        except Exception:
            return None

    def _parse_epw(self, epw_path: str, city_name: Optional[str] = None) -> ClimateSnapshot:
        cache_key = f"{os.path.abspath(epw_path)}::{city_name or ''}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        fallback_city = city_name or os.path.splitext(os.path.basename(epw_path))[0]

        latitude = None
        longitude = None
        elevation_m = None
        db_values = []
        rh_values = []
        winter_db_values = []
        summer_db_values = []
        hdd18 = 0.0
        cdd26 = 0.0

        with open(epw_path, "r", encoding="utf-8", errors="replace") as f:
            header = f.readline().strip().split(",")
            # EPW LOCATION: LOCATION,City,State,Country,Data Source,WMO,Lat,Lon,TZ,Elevation
            if len(header) >= 10 and str(header[0]).strip().upper() == "LOCATION":
                header_city = str(header[1]).strip()
                if header_city:
                    fallback_city = city_name or header_city
                latitude = self._safe_float(header[6])
                longitude = self._safe_float(header[7])
                elevation_m = self._safe_float(header[9])

            for _ in range(7):
                f.readline()

            for line in f:
                cols = line.strip().split(",")
                if len(cols) < 9:
                    continue

                month = self._safe_float(cols[1])
                db = self._safe_float(cols[6])  # Dry Bulb Temperature
                rh = self._safe_float(cols[8])  # Relative Humidity
                if db is None:
                    continue

                db_values.append(db)
                if rh is not None:
                    rh_values.append(rh)

                if month in (12.0, 1.0, 2.0):
                    winter_db_values.append(db)
                elif month in (6.0, 7.0, 8.0):
                    summer_db_values.append(db)

                hdd18 += max(0.0, 18.0 - db) / 24.0
                cdd26 += max(0.0, db - 26.0) / 24.0

        def _mean(values):
            if not values:
                return None
            return sum(values) / len(values)

        snapshot = ClimateSnapshot(
            city_name=fallback_city,
            latitude=latitude,
            longitude=longitude,
            elevation_m=elevation_m,
            annual_mean_db_c=_mean(db_values),
            annual_mean_rh_pct=_mean(rh_values),
            annual_hdd18_c_day=hdd18,
            annual_cdd26_c_day=cdd26,
            winter_mean_db_c=_mean(winter_db_values),
            summer_mean_db_c=_mean(summer_db_values),
        )
        self._cache[cache_key] = snapshot
        return snapshot

    def build_directive(self, epw_path: str, city_name: Optional[str] = None) -> str:
        """输出可直接拼接到LLM请求中的气候上下文提示。"""
        try:
            snapshot = self._parse_epw(epw_path=epw_path, city_name=city_name)
        except Exception as e:
            fallback_city = city_name or os.path.splitext(os.path.basename(epw_path))[0]
            return (
                "【气候适应性上下文】\n"
                f"- 城市: {fallback_city}\n"
                f"- 气候数据读取失败: {e}\n"
                "- 请基于可用信息自主判断该城市气候条件，不预设固定优化方向。"
            )

        def _fmt(v, digits=2, unit=""):
            if v is None:
                return "未知"
            return f"{float(v):.{digits}f}{unit}"

        hdd = snapshot.annual_hdd18_c_day
        cdd = snapshot.annual_cdd26_c_day
        hdd_cdd_ratio = None
        if hdd is not None and cdd is not None and cdd > 0:
            hdd_cdd_ratio = hdd / cdd

        return (
            "【气候适应性上下文】\n"
            f"- 城市: {snapshot.city_name}\n"
            f"- 位置: 纬度 {_fmt(snapshot.latitude, 3)}，经度 {_fmt(snapshot.longitude, 3)}，海拔 {_fmt(snapshot.elevation_m, 1, ' m')}\n"
            f"- 年均干球温度: {_fmt(snapshot.annual_mean_db_c, 2, ' C')}\n"
            f"- 年均相对湿度: {_fmt(snapshot.annual_mean_rh_pct, 1, ' %')}\n"
            f"- 采暖度日 HDD18: {_fmt(snapshot.annual_hdd18_c_day, 1)} C-day\n"
            f"- 制冷度日 CDD26: {_fmt(snapshot.annual_cdd26_c_day, 1)} C-day\n"
            f"- HDD18/CDD26 比值: {_fmt(hdd_cdd_ratio, 3)}\n"
            f"- 冬季均温(12-2月): {_fmt(snapshot.winter_mean_db_c, 2, ' C')}\n"
            f"- 夏季均温(6-8月): {_fmt(snapshot.summer_mean_db_c, 2, ' C')}\n"
            "- 说明: 以上仅提供气候与分区判读线索，不做“采暖/制冷主导”预判，"
            " 不预设“某城市必须改某字段”。"
            " 请结合当前迭代反馈与该城市气候条件，自主判断并选择预期收益更高且工程可行的对象与字段组合。"
        )
