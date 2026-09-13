"""车牌后处理规则：省份白名单、易混字符纠正、位数校验。纯函数可单测。

计划书 §13.2 面试点：省份白名单校验 / 位数校验 / 易混字符纠正（0/O、1/I）。
"""

from __future__ import annotations

import re

# 31 个省级行政区简称（不含港澳台特种牌）
PROVINCES = "京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼"

# 发牌机关代码位（车牌第 2 位）合法字母：去 I、O
LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"

# 后 5 位（蓝牌）合法字符：字母（去 I、O）+ 数字
BODY_CHARS = set(LETTERS) | set("0123456789")

# 新能源牌（绿牌）第 2 位专用字母：D 纯电 / F 混动
NEV_LETTERS = "DF"


def correct_confusable(plate: str) -> str:
    """纠正易混字符。

    - 第 2 位（发牌机关代码）必须是字母：0→O、1→I。
    - 后 5 位中 O→0、I→1（中国车牌字母表不含 I、O）。
    """
    if not plate:
        return plate
    chars = list(plate)
    if len(chars) >= 2:
        if chars[1] == "0":
            chars[1] = "O"
        elif chars[1] == "1":
            chars[1] = "I"
    for i in range(2, len(chars)):
        if chars[i] == "O":
            chars[i] = "0"
        elif chars[i] == "I":
            chars[i] = "1"
    return "".join(chars)


def validate_plate(plate: str) -> tuple[bool, str]:
    """校验车牌合法性。返回 (是否合法, 原因)。

    规则：
      1. 首位必须是省份简称。
      2. 第 2 位必须是发牌机关字母。
      3. 位数：8 位 = 新能源绿牌（第 3 位 D/F）；7 位 = 传统牌。
      4. 后段字符在合法字符集内。
    """
    if not plate:
        return False, "empty"

    plate = plate.strip().upper()

    if len(plate) < 2:
        return False, "too_short"

    if plate[0] not in PROVINCES:
        return False, "bad_province"

    if plate[1] not in LETTERS:
        return False, "bad_letter"

    # 位数：8 位 = 新能源绿牌，7 位 = 传统牌
    # 新能源 D/F 位置：小型车第 3 位（京AD12345），大型车末尾（京A12345D）
    if len(plate) == 8:
        if plate[2] in NEV_LETTERS:
            body = plate[3:]          # 小型新能源：D/F 在第 3 位
        elif plate[-1] in NEV_LETTERS:
            body = plate[2:-1]        # 大型新能源：D/F 在末尾
        else:
            return False, "nev_bad_letter"
    elif len(plate) == 7:
        body = plate[2:]
    else:
        return False, "wrong_len"

    if any(c not in BODY_CHARS for c in body):
        return False, "bad_body_char"

    return True, "ok"


def extract_plate_candidate(text: str) -> str:
    """从一段 OCR 文本中提取最像车牌号的子串（兜底清洗）。

    匹配「省份简称 + 字母 + 5~6 位字母数字」，返回首个候选。
    """
    m = re.search(rf"[{PROVINCES}][A-Z][A-Z0-9]{{5,6}}", text.upper())
    return m.group(0) if m else ""
