"""从 ultralytics 的指标对象里抽取可直接上报的数值。

``train.py`` 与 ``val.py`` 都会用到，所以放在这里共享。
"""

from __future__ import annotations

from typing import Any


def per_class_payload(box: Any, names: Any) -> list[dict[str, Any]]:
    """把逐类 mAP 转成 ``[{"class_id", "name", "mAP50-95"}]``。

    ``box.maps`` 是**按类别 id 索引**的定长数组（长度等于 ``nc``），
    而 ``box.ap_class_index`` 才是"哪些类别真的有样本"。
    因此必须用 ``maps[class_id]`` 取值，用列表下标会整体错位。
    """
    try:
        maps = [float(v) for v in box.maps]
    except (AttributeError, TypeError):
        return []
    try:
        ap_class_index = [int(v) for v in box.ap_class_index]
    except (AttributeError, TypeError):
        return []

    name_table = names if isinstance(names, dict) else {}
    payload: list[dict[str, Any]] = []
    for class_id in ap_class_index:
        if not 0 <= class_id < len(maps):
            continue
        payload.append(
            {
                "class_id": class_id,
                "name": str(name_table.get(class_id, f"class_{class_id}")),
                "mAP50-95": round(maps[class_id], 6),
            }
        )
    return payload
