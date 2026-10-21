"""
人工标注的分发与一致性：为每位标注者写一份打乱顺序、去掉线索的标签表，再读回
各自的标签，计算 Cohen's κ（两两）、Fleiss' κ（三人及以上）、一致率和混淆矩阵。
Distributing human labelling and measuring agreement: write each labeller a
shuffled copy of the label sheet with the hints removed, read their labels
back, and compute Cohen's kappa (per pair), Fleiss' kappa (three or more
labellers), percent agreement and the confusion matrix.

标注者的表只有 item_id（不透明的哈希）、用户请求、工具名和参数：run_id 是
episode 文件夹名，会暴露攻击编号和 setup；ground truth 预标注会影响判断。
A labeller's copy only has item_id (an opaque hash), the user request, the tool
name and its arguments: run_id is the episode folder name, which reveals the
injection task and setup, and ground-truth pre-labels would sway the labeller.
"""

import csv
import hashlib
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

from .reviewer_errors import SAFE, UNSAFE, VALID_LABELS, _verdict

CATEGORIES = (SAFE, UNSAFE)
# 标注者表的列；comment 留给标注者写理由。
# Columns of a labeller's copy; comment is for the labeller's reasoning.
LABELLER_COLUMNS = (
    "item_id",
    "user_request",
    "tool_name",
    "arguments",
    "label",
    "comment",
)


def item_id(run_id: Any, tool_call_id: Any) -> str:
    """
    一个被审核调用的不透明编号；同一调用每次生成都相同，不暴露 run_id。
    An opaque id for one reviewed call; the same call always gets the same id,
    and it reveals nothing about run_id.
    """

    digest = hashlib.sha256(f"{run_id}\x00{tool_call_id}".encode("utf-8"))
    return "c" + digest.hexdigest()[:10]


def labeller_rows(
    rows: Iterable[Mapping[str, Any]],
    labeller: str,
    seed: int,
    include_prelabelled: bool = False,
) -> list[dict[str, Any]]:
    """
    一位标注者的表：默认只含需要人工判断的行，每人一个由 (seed, 姓名) 决定的顺序。
    One labeller's copy: by default only the rows that need a human, in an
    order fixed by (seed, name), different for each labeller.
    """

    selected = [
        {
            "item_id": row["item_id"],
            "user_request": row.get("user_request", ""),
            "tool_name": row.get("tool_name", ""),
            "arguments": row.get("arguments", ""),
            "label": "",
            "comment": "",
        }
        for row in rows
        if include_prelabelled or not row.get("label")
    ]
    # 先按 item_id 排序，打乱结果只取决于 seed 和姓名，与输入顺序无关。
    # Sort by item_id first, so the shuffle depends only on the seed and name,
    # not on the input order.
    selected.sort(key=lambda row: row["item_id"])
    random.Random(f"{seed}:{labeller}").shuffle(selected)
    return selected


def write_labeller_sheet(rows: Iterable[Mapping[str, Any]], path: str | Path) -> int:
    """
    写出标注者的表；utf-8-sig 让 Excel 正确显示。
    Write a labeller's copy; utf-8-sig so Excel shows it correctly.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=LABELLER_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in LABELLER_COLUMNS})
            count += 1
    return count


@dataclass
class LabellerSheet:
    """
    读回的一份标注表：有效标签，以及空白、无效和重复的行（行号从表头后的 2 开始）。
    One labeller's sheet as read back: valid labels, plus blank, invalid and
    duplicate rows (row numbers start at 2, after the header).
    """

    name: str
    labels: dict[str, str] = field(default_factory=dict)
    blank: list[str] = field(default_factory=list)
    invalid: list[tuple[int, str]] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)


def read_labeller_sheet(path: str | Path, name: str) -> LabellerSheet:
    """
    读取标注者填好的表。无效值不报错，而是记录下来，便于一次性列出所有问题。
    Read a filled-in copy. Invalid values are recorded rather than raised, so
    every problem can be listed at once.
    """

    sheet = LabellerSheet(name=name)
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing = {"item_id", "label"} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{path} is missing columns: {', '.join(sorted(missing))}"
            )
        for row_number, row in enumerate(reader, start=2):
            item = (row.get("item_id") or "").strip()
            if not item:
                continue
            label = (row.get("label") or "").strip().casefold()
            if item in sheet.labels or item in sheet.blank:
                sheet.duplicates.append(item)
                continue
            if not label:
                sheet.blank.append(item)
            elif label not in VALID_LABELS:
                sheet.invalid.append((row_number, label))
            else:
                sheet.labels[item] = label
    return sheet


# ---------------------------------------------------------------- 一致性 / Agreement


def confusion_matrix(
    first: Sequence[str], second: Sequence[str]
) -> dict[str, dict[str, int]]:
    """
    matrix[a][b]：第一位标 a、第二位标 b 的条目数。
    matrix[a][b]: the number of items the first labeller marked a and the
    second marked b.
    """

    matrix = {a: {b: 0 for b in CATEGORIES} for a in CATEGORIES}
    for a, b in zip(first, second):
        matrix[a][b] += 1
    return matrix


def percent_agreement(first: Sequence[str], second: Sequence[str]) -> float | None:
    if not first:
        return None
    return sum(a == b for a, b in zip(first, second)) / len(first)


def cohen_kappa(first: Sequence[str], second: Sequence[str]) -> float | None:
    """
    Cohen's κ = (p_o − p_e) / (1 − p_e)。两人都只用了同一个类别时 p_e = 1，
    κ 没有定义，返回 None（此时看一致率）。
    Cohen's kappa = (p_o - p_e) / (1 - p_e). When both labellers used only one
    and the same category, p_e = 1 and kappa is undefined, so None is returned
    (look at percent agreement instead).
    """

    n = len(first)
    if n == 0:
        return None
    observed = percent_agreement(first, second)
    expected = sum(
        (first.count(category) / n) * (second.count(category) / n)
        for category in CATEGORIES
    )
    if expected == 1:
        return None
    return (observed - expected) / (1 - expected)


def fleiss_kappa(ratings: Sequence[Sequence[str]]) -> float | None:
    """
    Fleiss' κ；ratings 中每一项是同一条目的全部标注，人数必须相同。
    Fleiss' kappa; each entry of ratings holds all labels for one item, and
    every item must have the same number of labels.
    """

    if not ratings:
        return None
    raters = len(ratings[0])
    if raters < 2 or any(len(item) != raters for item in ratings):
        raise ValueError("every item needs the same number of labels, at least 2")
    items = len(ratings)
    counts = [[item.count(category) for category in CATEGORIES] for item in ratings]
    agreement = sum(
        (sum(c * c for c in row) - raters) / (raters * (raters - 1)) for row in counts
    ) / items
    shares = [sum(row[j] for row in counts) / (items * raters) for j in range(len(CATEGORIES))]
    expected = sum(share * share for share in shares)
    if expected == 1:
        return None
    return (agreement - expected) / (1 - expected)


def compare(
    a_name: str,
    a_labels: Mapping[str, str],
    b_name: str,
    b_labels: Mapping[str, str],
    items: Iterable[str],
) -> dict[str, Any]:
    """
    两组标签在 items 中双方都有标签的条目上的 κ、一致率和混淆矩阵。
    Kappa, percent agreement and confusion matrix of two sets of labels, on the
    items in `items` that both of them labelled.
    """

    shared = [item for item in items if item in a_labels and item in b_labels]
    a = [a_labels[item] for item in shared]
    b = [b_labels[item] for item in shared]
    return {
        "a": a_name,
        "b": b_name,
        "items": len(shared),
        "percent_agreement": percent_agreement(a, b),
        "cohen_kappa": cohen_kappa(a, b),
        "confusion_matrix": confusion_matrix(a, b),
    }


def pairwise_agreement(
    sheets: Sequence[LabellerSheet], items: Sequence[str]
) -> list[dict[str, Any]]:
    """
    每对标注者在共同标注过的条目上的 κ、一致率和混淆矩阵。
    Kappa, percent agreement and confusion matrix for each pair of labellers,
    on the items both of them labelled.
    """

    return [
        compare(first.name, first.labels, second.name, second.labels, items)
        for first, second in combinations(sheets, 2)
    ]


def reviewer_labels(
    reviews: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, str], set[Any]]:
    """
    把模拟 reviewer 的结论当作一组标签：ALLOW 为 safe，DENY 为 unsafe，按
    item_id 对应。返回标签和日志中出现的 reviewer 模型。失败的审核应在读取时
    已被跳过（reviewer_errors.read_reviewer_log）。
    The simulated reviewer's verdicts as a set of labels: ALLOW is safe, DENY is
    unsafe, keyed by item_id. Returns the labels and the reviewer models found
    in the logs. Failed reviews should already be skipped when reading
    (reviewer_errors.read_reviewer_log).
    """

    labels: dict[str, str] = {}
    models: set[Any] = set()
    for review in reviews:
        verdict = _verdict(review.get("reviewer_verdict"))
        if verdict is None:
            continue
        labels[item_id(review.get("run_id"), review.get("tool_call_id"))] = (
            SAFE if verdict == "ALLOW" else UNSAFE
        )
        models.add(review.get("reviewer_model"))
    return labels, models


@dataclass
class SettledSheet:
    """
    读回的 disagreements.csv：讨论后的 final_label，以及空白和无效的行。
    disagreements.csv read back: the settled final_label values, plus blank and
    invalid rows.
    """

    labels: dict[str, str] = field(default_factory=dict)
    blank: list[str] = field(default_factory=list)
    invalid: list[tuple[int, str]] = field(default_factory=list)


def read_settled(path: str | Path) -> SettledSheet:
    settled = SettledSheet()
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing = {"item_id", "final_label"} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{path} is missing columns: {', '.join(sorted(missing))}"
            )
        for row_number, row in enumerate(reader, start=2):
            item = (row.get("item_id") or "").strip()
            label = (row.get("final_label") or "").strip().casefold()
            if not item:
                continue
            if not label:
                settled.blank.append(item)
            elif label not in VALID_LABELS:
                settled.invalid.append((row_number, label))
            else:
                settled.labels[item] = label
    return settled
