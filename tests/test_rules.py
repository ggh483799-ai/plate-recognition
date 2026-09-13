"""后处理规则单元测试。计划书 §6 M3 / §9.3。"""

from src.postprocess.rules import (
    PROVINCES,
    correct_confusable,
    extract_plate_candidate,
    validate_plate,
)


class TestCorrectConfusable:
    def test_second_digit_letter(self):
        # 第 2 位必须是字母：0→O、1→I
        assert correct_confusable("京012345") == "京O12345"
        assert correct_confusable("京112345") == "京I12345"

    def test_body_letter_confusion(self):
        # 后段 O→0、I→1（车牌字母表不含 I/O）
        assert correct_confusable("京A12O45") == "京A12045"
        assert correct_confusable("京A12I45") == "京A12145"

    def test_no_change(self):
        assert correct_confusable("京A12345") == "京A12345"


class TestValidatePlate:
    def test_valid_blue_plate(self):
        ok, reason = validate_plate("京A12345")
        assert ok, reason

    def test_valid_nev_plate(self):
        # 新能源 8 位，第 2 位发牌机关字母，第 3 位 D/F（小型车）
        ok, reason = validate_plate("京AD12345")
        assert ok, reason
        ok2, _ = validate_plate("沪FD12345")
        assert ok2

    def test_valid_nev_large_plate(self):
        # 大型新能源牌：D/F 在末尾
        ok, reason = validate_plate("京A12345D")
        assert ok, reason

    def test_bad_province(self):
        ok, reason = validate_plate("X1234567")
        assert not ok
        assert reason == "bad_province"

    def test_bad_letter(self):
        ok, reason = validate_plate("京112345")
        assert not ok
        assert reason == "bad_letter"

    def test_wrong_len(self):
        ok, reason = validate_plate("京A1234")  # 6 位
        assert not ok
        assert reason == "wrong_len"

    def test_nev_wrong_len(self):
        ok, reason = validate_plate("京AD123")  # 6 位
        assert not ok
        assert reason == "wrong_len"

    def test_nev_bad_letter(self):
        # 8 位但第 3 位不是 D/F
        ok, reason = validate_plate("京AA12345")
        assert not ok
        assert reason == "nev_bad_letter"

    def test_bad_body_char(self):
        ok, reason = validate_plate("京A12O45")  # 含字母 O
        assert not ok
        assert reason == "bad_body_char"


class TestExtractCandidate:
    def test_extract_from_text(self):
        assert extract_plate_candidate("车辆京A12345通过") == "京A12345"

    def test_no_plate(self):
        assert extract_plate_candidate("hello world") == ""


def test_province_count():
    # 31 个省级行政区简称
    assert len(PROVINCES) == 31
