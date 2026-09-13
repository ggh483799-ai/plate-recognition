"""透视矫正单元测试。计划书 §6 M3 / §9.3。"""

import numpy as np

from src.vision.warp import PLATE_H, PLATE_W, order_points, pad_bbox, warp_plate


class TestOrderPoints:
    def test_sorted_input(self):
        pts = np.array([[0, 0], [94, 0], [94, 24], [0, 24]], dtype=np.float32)
        ordered = order_points(pts)
        assert np.allclose(ordered[0], [0, 0])   # 左上
        assert np.allclose(ordered[1], [94, 0])  # 右上
        assert np.allclose(ordered[2], [94, 24])  # 右下
        assert np.allclose(ordered[3], [0, 24])  # 左下

    def test_shuffled_input(self):
        pts = np.array([[94, 24], [0, 0], [0, 24], [94, 0]], dtype=np.float32)
        ordered = order_points(pts)
        assert np.allclose(ordered[0], [0, 0])
        assert np.allclose(ordered[1], [94, 0])
        assert np.allclose(ordered[2], [94, 24])
        assert np.allclose(ordered[3], [0, 24])


class TestWarpPlate:
    def test_output_size(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        pts = np.array([[10, 10], [190, 10], [190, 90], [10, 90]], dtype=np.float32)
        warped = warp_plate(img, pts)
        assert warped.shape == (PLATE_H, PLATE_W, 3)


class TestPadBBox:
    def test_pad_ratio(self):
        pts = pad_bbox((100, 100, 300, 200), ratio=0.1)
        # 宽 200、高 100，各边外扩 10% → 20 / 10
        assert np.allclose(pts[0], [80, 90])    # 左上
        assert np.allclose(pts[2], [320, 210])  # 右下
