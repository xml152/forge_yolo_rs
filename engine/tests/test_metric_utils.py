"""``per_class_payload`` 的回归测试。

运行：``cd engine && .venv/bin/python -m unittest discover -s tests -v``
"""

from __future__ import annotations

import unittest

from engine.metric_utils import per_class_payload


class _Box:
    """复刻 ultralytics ``Metric`` 的 ``maps`` / ``ap_class_index`` 语义。"""

    def __init__(self, nc: int, ap: list[float], ap_class_index: list[int], mean: float) -> None:
        self.nc = nc
        self.ap = ap
        self.ap_class_index = ap_class_index
        self.map = mean

    @property
    def maps(self) -> list[float]:
        # 与 ultralytics 一致：定长 nc，按类别 id 写入，未命中的类别填平均值。
        maps = [self.map] * self.nc
        for position, class_id in enumerate(self.ap_class_index):
            maps[class_id] = self.ap[position]
        return maps


_NAMES = {0: "person", 1: "car", 3: "dog"}


class PerClassPayloadTests(unittest.TestCase):
    def test_all_classes_present(self) -> None:
        box = _Box(3, [0.1, 0.2, 0.3], [0, 1, 2], 0.0)
        got = per_class_payload(box, _NAMES)
        self.assertEqual([row["mAP50-95"] for row in got], [0.1, 0.2, 0.3])
        self.assertEqual([row["class_id"] for row in got], [0, 1, 2])

    def test_missing_classes_do_not_shift_values(self) -> None:
        """只有 1/3 类有样本时，不能按列表下标取值（会整体错位）。"""
        box = _Box(4, [0.10, 0.30], [1, 3], 0.0)
        got = per_class_payload(box, _NAMES)
        self.assertEqual([(row["class_id"], row["mAP50-95"]) for row in got], [(1, 0.10), (3, 0.30)])

    def test_names_are_resolved_by_class_id(self) -> None:
        box = _Box(4, [0.10, 0.30], [1, 3], 0.0)
        got = per_class_payload(box, _NAMES)
        self.assertEqual([row["name"] for row in got], ["car", "dog"])

    def test_missing_name_falls_back_to_class_id(self) -> None:
        box = _Box(2, [0.5], [1], 0.0)
        self.assertEqual(per_class_payload(box, {})[0]["name"], "class_1")

    def test_absent_attributes_yield_empty_payload(self) -> None:
        self.assertEqual(per_class_payload(object(), None), [])

    def test_out_of_range_class_id_is_skipped(self) -> None:
        class _Inconsistent:
            maps = [0.5]
            ap_class_index = [0, 7]

        got = per_class_payload(_Inconsistent(), _NAMES)
        self.assertEqual([row["class_id"] for row in got], [0])


if __name__ == "__main__":
    unittest.main()
