"""视频流水线测试：去重语义、落盘、统计、画框、读取器协议。计划书 §9.3。

全部用注入的 fake 帧源与 fake pipeline，不加载真实模型，保证毫秒级、可离线跑。
"""

import csv
import json

import numpy as np
import pytest

from src.common.interfaces import PlateResult
from src.io.reader import SingleFrameReader, VideoReader, iter_frames, make_reader
from src.io.writer import AnnotatedVideoWriter
from src.pipeline.video_pipeline import (
    CSV_FIELDS,
    CsvEventSink,
    JsonlEventSink,
    PlateDeduplicator,
    ShotWriter,
    VideoPipeline,
)
from src.vision.draw import crop_plate, draw_detections


# ============================================================
# 测试替身
# ============================================================

class FakeFrameReader:
    """按预设帧序列产出，并记录 close 是否被调用。"""

    def __init__(self, frames, fps=25.0):
        self._frames = list(frames)
        self.fps = fps
        self.closed = False

    def read(self):
        if not self._frames:
            return np.empty((0, 0, 3), dtype=np.uint8)
        return self._frames.pop(0)

    def close(self):
        self.closed = True


class FakePipeline:
    """按调用次序返回预设结果；超出预设则返回空。"""

    def __init__(self, per_frame):
        self.per_frame = list(per_frame)
        self.calls = 0

    def run(self, frame):
        idx = self.calls
        self.calls += 1
        return self.per_frame[idx] if idx < len(self.per_frame) else []


class CountingWriter:
    """只计数不编码，避免测试依赖视频编码器。"""

    def __init__(self):
        self.frames = []
        self.closed = False
        self.path = "fake.mp4"

    def write(self, frame):
        self.frames.append(frame)

    def close(self):
        self.closed = True


def _plate(no="粤A3333G", rec=0.99, det=0.86, color="blue"):
    return PlateResult(
        plate_no=no, plate_color=color, vehicle_type="car",
        det_score=det, rec_score=rec, bbox=[10, 20, 110, 60], cost_ms=12.0,
    )


def _frames(n, size=(80, 120)):
    return [np.zeros((size[0], size[1], 3), dtype=np.uint8) for _ in range(n)]


# ============================================================
# 去重语义
# ============================================================

def test_dedup_merges_within_window():
    dedup = PlateDeduplicator(window_s=3.0)
    assert len(dedup.accept([_plate()], 0, 0.0)) == 1   # 首现
    assert dedup.accept([_plate()], 1, 0.04) == []      # 窗内 → 不算新事件
    assert dedup.accept([_plate()], 2, 0.08) == []

    events = dedup.events
    assert len(events) == 1
    assert events[0].hits == 3
    assert events[0].frame_idx == 0
    assert dedup.total_hits == 3


def test_dedup_new_event_after_window():
    dedup = PlateDeduplicator(window_s=1.0)
    dedup.accept([_plate()], 0, 0.0)
    dedup.accept([_plate()], 1, 0.5)                  # 窗内
    fresh = dedup.accept([_plate()], 5, 4.0)          # 间隔 3.5s > 1s → 车已离开，算新事件
    assert len(fresh) == 1
    assert len(dedup.events) == 2
    assert [e.hits for e in dedup.events] == [2, 1]


def test_dedup_keeps_best_confidence():
    # 同一车牌在不同帧清晰度不同：上报值应取识别置信度最高的那一次
    dedup = PlateDeduplicator(window_s=5.0)
    dedup.accept([_plate(rec=0.60, det=0.70)], 0, 0.0)
    dedup.accept([_plate(rec=0.98, det=0.90)], 1, 0.04)
    dedup.accept([_plate(rec=0.75, det=0.80)], 2, 0.08)

    ev = dedup.events[0]
    assert ev.hits == 3
    assert ev.rec_score == pytest.approx(0.98)
    assert ev.det_score == pytest.approx(0.90)


def test_dedup_separates_plates_and_skips_empty():
    dedup = PlateDeduplicator(window_s=5.0)
    results = [_plate(no="粤A3333G"), _plate(no="京B11111"), PlateResult(plate_no="")]
    fresh = dedup.accept(results, 0, 0.0)
    assert sorted(e.plate_no for e in fresh) == ["京B11111", "粤A3333G"]
    assert dedup.total_hits == 2          # 空车牌号不计入命中
    assert dedup.active_count == 2


def test_dedup_rejects_negative_window():
    with pytest.raises(ValueError):
        PlateDeduplicator(window_s=-1.0)


def test_dedup_summary_by_color():
    dedup = PlateDeduplicator(window_s=1.0)
    dedup.accept([_plate(no="粤A3333G", color="blue")], 0, 0.0)
    dedup.accept([_plate(no="粤BD12345", color="green")], 1, 0.1)
    s = dedup.summary()
    assert s["events"] == 2
    assert s["by_plate_color"] == {"blue": 1, "green": 1}
    assert s["window_s"] == 1.0


def test_dedup_event_is_pending_until_plate_leaves():
    """事件在车牌离场前不落盘：pop_closed 只在「超窗未再出现」后吐出定稿事件。"""
    dedup = PlateDeduplicator(window_s=1.0)
    dedup.accept([_plate()], 0, 0.0)
    assert dedup.pop_closed() == []            # 刚出现，未定稿

    dedup.accept([_plate()], 1, 0.5)
    assert dedup.pop_closed() == []            # 仍在窗内，未定稿

    dedup.accept([], 2, 2.0)                   # 2.0 - 0.5 = 1.5 > 1.0 → 判为离场
    closed = dedup.pop_closed()
    assert len(closed) == 1
    assert closed[0].hits == 2                 # 定稿事件带完整命中次数
    assert dedup.pop_closed() == []            # 只吐一次
    assert dedup.active_count == 0


def test_dedup_flush_closes_active_events():
    dedup = PlateDeduplicator(window_s=99.0)
    dedup.accept([_plate()], 0, 0.0)            # 窗口很长，车「还在场」
    assert dedup.pop_closed() == []
    closed = dedup.flush()                      # 流结束 → 强制定稿
    assert len(closed) == 1
    assert closed[0].hits == 1
    assert dedup.active_count == 0
    assert dedup.flush() == []


def test_dedup_new_event_after_leave_has_own_hits():
    dedup = PlateDeduplicator(window_s=1.0)
    dedup.accept([_plate()], 0, 0.0)
    dedup.accept([_plate()], 1, 0.4)
    dedup.accept([], 2, 2.0)                    # 离场 → 第 1 个事件定稿
    dedup.accept([_plate()], 3, 2.1)            # 重现 → 第 2 个事件
    assert [e.hits for e in dedup.pop_closed()] == [2]
    assert [e.hits for e in dedup.flush()] == [1]


# ============================================================
# 编排与统计
# ============================================================

def test_process_stats_and_events():
    reader = FakeFrameReader(_frames(5), fps=25.0)
    pipeline = FakePipeline([[_plate()], [], [_plate()], [], []])
    vp = VideoPipeline(pipeline, window_s=3.0)

    stats = vp.process(reader, log_every=0)

    assert stats["frames"] == 5
    assert stats["frames_with_plate"] == 2
    assert stats["events"] == 1                  # 5 帧里同车牌只算 1 个事件
    assert stats["new_events"] == 1
    assert stats["total_hits"] == 2
    assert stats["distinct_plates"] == 1
    assert stats["source_fps"] == 25.0
    assert reader.closed is True                 # iter_frames 负责释放
    assert pipeline.calls == 5


def test_process_max_frames_limit():
    reader = FakeFrameReader(_frames(10))
    vp = VideoPipeline(FakePipeline([]), window_s=1.0)
    stats = vp.process(reader, max_frames=3, log_every=0)
    assert stats["frames"] == 3
    assert reader.closed is True


def test_process_writes_annotated_frames():
    reader = FakeFrameReader(_frames(4))
    writer = CountingWriter()
    vp = VideoPipeline(FakePipeline([[_plate()], [], [], []]), window_s=1.0)
    stats = vp.process(reader, writer=writer, log_every=0)

    assert len(writer.frames) == 4               # 每帧都写（含无车牌帧）
    assert writer.closed is True
    assert stats["video_frames_written"] == 0    # CountingWriter 无 frames_written，走默认分支


def test_process_uses_frame_timeline_when_fps_known():
    # fps=10 → 第 3 帧时间轴 0.2s；两帧间隔 0.2s < 窗口 → 仍是同一事件
    reader = FakeFrameReader(_frames(3), fps=10.0)
    vp = VideoPipeline(FakePipeline([[_plate()], [_plate()], [_plate()]]), window_s=1.0)
    stats = vp.process(reader, log_every=0)
    assert stats["events"] == 1
    assert stats["total_hits"] == 3


def test_process_timeline_ignores_warmup_when_fps_unknown():
    """无帧率来源（单图/未知流）时，首帧时间戳不能被模型预热耗时污染。"""
    import time

    class SlowPipeline:
        def run(self, frame):
            time.sleep(0.05)      # 模拟模型预热/推理耗时
            return [_plate()]

    reader = FakeFrameReader(_frames(1), fps=None)   # 图片场景：无帧率
    vp = VideoPipeline(SlowPipeline(), window_s=5.0)
    vp.process(reader, log_every=0)

    assert vp.dedup.events[0].t_sec < 0.05           # 而不是 0.05+（旧实现会是整个预热时长）


# ============================================================
# 落盘
# ============================================================

def test_csv_sink_content_and_bom(tmp_path):
    path = tmp_path / "detections.csv"
    with CsvEventSink(path) as sink:
        sink.write(PlateDeduplicator().accept([_plate()], 7, 1.25)[0])

    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")       # Excel 中文不乱码的必要条件
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    assert rows[0]["plate_no"] == "粤A3333G"
    assert rows[0]["frame_idx"] == "7"
    assert rows[0]["t_sec"] == "1.25"
    assert set(rows[0]) == set(CSV_FIELDS)


def test_jsonl_sink_content(tmp_path):
    path = tmp_path / "events.jsonl"
    dedup = PlateDeduplicator(window_s=5.0)
    with JsonlEventSink(path) as sink:
        for ev in dedup.accept([_plate(), _plate(no="京B11111", color="green")], 3, 0.12):
            sink.write(ev)

    lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    assert {x["plate_no"] for x in lines} == {"粤A3333G", "京B11111"}
    assert lines[0]["bbox"] == [10.0, 20.0, 110.0, 60.0]


def test_process_reports_finalized_event_with_best_frame(tmp_path):
    """落盘的是「定稿事件」：hits 完整、取值取置信度最高的那一帧。"""
    better = PlateResult(
        plate_no="粤A3333G", plate_color="blue", vehicle_type="car",
        det_score=0.91, rec_score=0.98, bbox=[12, 22, 112, 62],
    )
    worse = PlateResult(
        plate_no="粤A3333G", plate_color="blue", vehicle_type="truck",
        det_score=0.70, rec_score=0.55, bbox=[8, 18, 108, 58],
    )
    path = tmp_path / "finalized.csv"
    reader = FakeFrameReader(_frames(6), fps=1.0)          # 1 fps → 帧号即秒数
    pipeline = FakePipeline([[worse], [better], [], [], [], []])
    vp = VideoPipeline(pipeline, window_s=1.0)

    with CsvEventSink(path) as sink:
        stats = vp.process(reader, sinks=[sink], log_every=0)

    assert stats["events"] == 1
    assert stats["new_events"] == 1
    assert stats["events_reported"] == 1
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig", newline="")))
    assert len(rows) == 1
    assert rows[0]["hits"] == "2"
    assert rows[0]["rec_score"] == "0.98"      # 不是首帧的 0.55
    assert rows[0]["vehicle_type"] == "car"    # 不是首帧误判的 truck


def test_process_writes_sinks_at_close(tmp_path):
    csv_path, jsonl_path = tmp_path / "d.csv", tmp_path / "d.jsonl"
    reader = FakeFrameReader(_frames(4), fps=25.0)
    pipeline = FakePipeline([[_plate()], [], [_plate()], []])
    vp = VideoPipeline(pipeline, window_s=3.0)

    with CsvEventSink(csv_path) as csv_sink, JsonlEventSink(jsonl_path) as jsonl_sink:
        vp.process(reader, sinks=[csv_sink, jsonl_sink], log_every=0)

    assert len(csv_path.read_text(encoding="utf-8-sig").splitlines()) == 2   # header + 1 行
    assert len(jsonl_path.read_text(encoding="utf-8").splitlines()) == 1


def test_process_closes_sinks_on_crash(tmp_path):
    """pipeline 抛异常时，落盘文件仍应被正常关闭（flush 不丢）。"""
    class BoomPipeline:
        def __init__(self):
            self.calls = 0

        def run(self, frame):
            self.calls += 1
            if self.calls >= 2:
                raise RuntimeError("boom")
            return [_plate()]

    path = tmp_path / "oncrash.csv"
    sink = CsvEventSink(path)
    reader = FakeFrameReader(_frames(5))
    with pytest.raises(RuntimeError):
        VideoPipeline(BoomPipeline(), window_s=1.0).process(reader, sinks=[sink], log_every=0)

    assert sink._fh.closed is True
    assert len(path.read_text(encoding="utf-8-sig").splitlines()) == 2


# ============================================================
# 画框
# ============================================================

def test_draw_detections_does_not_mutate_input():
    frame = np.zeros((200, 400, 3), dtype=np.uint8)
    out = draw_detections(frame, [_plate()])
    assert out is not frame
    assert frame.sum() == 0
    assert out.sum() > 0


def test_draw_detections_inplace_and_empty():
    frame = np.zeros((200, 400, 3), dtype=np.uint8)
    out = draw_detections(frame, [_plate()], inplace=True)
    assert out is frame
    assert frame.sum() > 0

    clean = np.zeros((50, 50, 3), dtype=np.uint8)
    assert draw_detections(clean, []).sum() == 0


def test_draw_detections_skips_invalid_bbox():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    bad = PlateResult(plate_no="粤A3333G", bbox=[10, 10, 10, 10])       # 零面积
    outside = PlateResult(plate_no="粤A3333G", bbox=[500, 500, 900, 900])  # 完全越界
    assert draw_detections(frame, [bad, outside]).sum() == 0


def test_draw_detections_hud_and_ascii_fallback():
    frame = np.zeros((120, 320, 3), dtype=np.uint8)
    out = draw_detections(frame, [_plate()], hud=["frame 1", "plate 1"], inplace=True)
    assert out.sum() > 0        # HUD 已绘制


# ============================================================
# 读取器协议与视频写出
# ============================================================

def test_single_frame_reader_yields_once():
    reader = SingleFrameReader(np.zeros((10, 10, 3), dtype=np.uint8))
    assert reader.fps is None
    assert reader.read().size == 300
    assert reader.read().size == 0
    reader.close()


def test_iter_frames_closes_reader():
    reader = FakeFrameReader(_frames(2))
    assert len(list(iter_frames(reader))) == 2
    assert reader.closed is True


def test_make_reader_image_vs_missing_path(tmp_path):
    from src.io.reader import imwrite_bgr

    img_path = tmp_path / "car.png"
    assert imwrite_bgr(img_path, np.zeros((10, 20, 3), dtype=np.uint8))
    reader = make_reader(str(img_path))
    assert isinstance(reader, SingleFrameReader)
    assert reader.read().shape[:2] == (10, 20)

    with pytest.raises(RuntimeError):
        make_reader(str(tmp_path / "not_a_video.mp4"))


def test_annotated_video_writer_real_encode(tmp_path):
    path = tmp_path / "out.mp4"
    writer = AnnotatedVideoWriter(path, fps=5.0)
    for i in range(3):
        writer.write(np.full((48, 64, 3), i * 60, dtype=np.uint8))
    writer.close()

    assert writer.frames_written == 3
    assert path.stat().st_size > 0
    assert writer.is_open is False


def test_annotated_video_writer_ignores_empty_frame(tmp_path):
    writer = AnnotatedVideoWriter(tmp_path / "out.mp4", fps=5.0)
    writer.write(np.empty((0, 0, 3), dtype=np.uint8))
    assert writer.frames_written == 0
    assert writer.is_open is False   # 空帧不应触发创建
    writer.close()


def test_annotated_video_writer_fixes_size_mismatch(tmp_path):
    """尺寸不符的帧必须被纠正后写入，不能被静默丢弃（实测过 OpenCV 会丢帧）。"""
    import cv2

    path = tmp_path / "mismatch.mp4"
    writer = AnnotatedVideoWriter(path, fps=5.0)
    writer.write(np.zeros((48, 64, 3), dtype=np.uint8))   # 首帧决定尺寸
    writer.write(np.zeros((40, 40, 3), dtype=np.uint8))   # 尺寸不符 → 应被纠正而非丢弃
    writer.close()

    assert writer.frames_written == 2
    assert writer.frames_resized == 1

    cap = cv2.VideoCapture(str(path))
    count = 0
    while cap.read()[0]:
        count += 1
    cap.release()
    assert count == 2                                     # 容器里确实有 2 帧


def test_video_reader_rejects_bad_source(tmp_path):
    with pytest.raises(RuntimeError):
        VideoReader(str(tmp_path / "nope.mp4"))


# ============================================================
# 最佳帧截图（服务端视频任务依赖）
# ============================================================

def test_dedup_reports_best_updates_only_on_new_best():
    dedup = PlateDeduplicator(window_s=5.0)
    dedup.accept([_plate(rec=0.60)], 0, 0.0)
    assert len(dedup.pop_best_updates()) == 1        # 首帧即当前最佳帧
    assert dedup.pop_best_updates() == []            # 只吐一次

    dedup.accept([_plate(rec=0.50)], 1, 0.04)        # 更差 → 不是最佳帧
    assert dedup.pop_best_updates() == []

    dedup.accept([_plate(rec=0.95)], 2, 0.08)        # 更好 → 最佳帧更新
    updated = dedup.pop_best_updates()
    assert len(updated) == 1
    assert updated[0].rec_score == pytest.approx(0.95)


def test_crop_plate_returns_independent_copy():
    """必须 copy：VideoCapture 复用帧缓冲，切片视图会被下一帧覆盖。"""
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[40:60, 50:150] = 255
    crop = crop_plate(frame, [50, 40, 150, 60])
    assert crop is not None and crop.sum() > 0

    frame[:] = 0                                     # 模拟底层缓冲被复用
    assert crop.sum() > 0                            # 没 copy 的话这里会变成 0


def test_crop_plate_margin_and_invalid_boxes():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    tight = crop_plate(frame, [40, 40, 60, 60], margin=0.0)
    loose = crop_plate(frame, [40, 40, 60, 60], margin=0.5)
    assert loose.shape[0] > tight.shape[0] and loose.shape[1] > tight.shape[1]

    assert crop_plate(frame, [60, 60, 40, 40]) is None        # 坐标反了
    assert crop_plate(frame, [10, 10, 10, 10]) is None        # 零面积
    assert crop_plate(frame, []) is None
    assert crop_plate(frame, [500, 500, 600, 600]) is None    # 完全越界
    assert crop_plate(np.empty((0, 0, 3), dtype=np.uint8), [1, 1, 5, 5]) is None


def test_shot_writer_saves_and_frees_memory(tmp_path):
    ev = PlateDeduplicator().accept([_plate()], 0, 0.0)[0]
    ev.shot_img = np.full((20, 60, 3), 128, dtype=np.uint8)
    shots = ShotWriter(tmp_path / "shots")

    assert shots.save(ev) == "shot_001.jpg"          # 序号命名，不含中文
    assert (tmp_path / "shots" / "shot_001.jpg").is_file()
    assert ev.shot == "shot_001.jpg"
    assert ev.shot_img is None                       # 落盘后释放，长流不涨内存
    assert shots.saved == 1
    assert ev.to_dict()["shot"] == "shot_001.jpg"    # 上报字段带上


def test_shot_writer_skips_event_without_image(tmp_path):
    ev = PlateDeduplicator().accept([_plate()], 0, 0.0)[0]
    assert ShotWriter(tmp_path / "s").save(ev) == ""
    assert ev.shot == ""


def test_process_saves_shots_and_reports_progress(tmp_path):
    reader = FakeFrameReader(_frames(4), fps=4.0)
    pipeline = FakePipeline([[_plate(rec=0.50)], [_plate(rec=0.90)], [], []])
    shots = ShotWriter(tmp_path / "shots")
    progress = []
    vp = VideoPipeline(pipeline, window_s=0.5)

    stats = vp.process(reader, shots=shots, log_every=0,
                       progress_cb=lambda f, t, d: progress.append((f, t, d)))

    assert stats["shots_saved"] == 1                 # 只裁"最佳那一帧"，不是每帧都存
    assert stats["events"] == 1
    assert (tmp_path / "shots" / "shot_001.jpg").is_file()
    assert [p[0] for p in progress] == [1, 2, 3, 4]  # 每帧回调一次，帧数单调递增
    assert [p[2] for p in progress] == [1, 2, 2, 2]  # 前两帧有车牌，累计计数不回落
    assert all(p[1] == 0 for p in progress)          # FakeFrameReader 不含 frame_count


def test_process_no_shots_when_writer_absent(tmp_path):
    """没给 ShotWriter 就不该裁图（省掉无用的内存拷贝与分配）。"""
    reader = FakeFrameReader(_frames(2), fps=2.0)
    vp = VideoPipeline(FakePipeline([[_plate()], [_plate()]]), window_s=5.0)
    stats = vp.process(reader, log_every=0)
    assert stats["shots_saved"] == 0
    assert vp.dedup.events[0].shot_img is None


def test_process_marks_truncated_only_when_source_is_longer():
    class CountedReader(FakeFrameReader):
        frame_count = 10

    reader = CountedReader(_frames(10))
    vp = VideoPipeline(FakePipeline([]), window_s=1.0)
    stats = vp.process(reader, max_frames=3, log_every=0)
    assert stats["frames"] == 3
    assert stats["total_frames"] == 10
    assert stats["truncated"] is True                # 没看全，必须如实说

    reader2 = CountedReader(_frames(10))
    stats2 = VideoPipeline(FakePipeline([]), window_s=1.0).process(reader2, log_every=0)
    assert stats2["frames"] == 10 and stats2["truncated"] is False


def test_video_reader_frame_count_matches_decoded(tmp_path):
    """容器声称的帧数应与可解码帧数一致（进度条百分比的可信度就建立在这上面）。"""
    path = tmp_path / "counted.mp4"
    writer = AnnotatedVideoWriter(path, fps=5.0)
    for _ in range(5):
        writer.write(np.zeros((32, 48, 3), dtype=np.uint8))
    writer.close()

    reader = VideoReader(str(path))
    claimed = reader.frame_count
    decoded = 0
    while reader.read().size:
        decoded += 1
    reader.close()

    assert decoded == 5
    assert claimed == decoded
