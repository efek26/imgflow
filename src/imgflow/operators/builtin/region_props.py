"""Bağlı bileşenler üzerinde temel bölge analizi: alan, çevre, centroid, bbox, dairesellik."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from imgflow.core.params import ParamSpec, ParamType
from imgflow.core.types import PortSpec, PortType
from imgflow.operators import registry


@registry.register
class RegionPropsOp:
    id = "analysis.region_props"
    inputs = [PortSpec("labels", PortType.IMAGE)]
    outputs = [PortSpec("measurements", PortType.MEASUREMENTS)]
    params = [
        ParamSpec("min_area", ParamType.FLOAT, default=0.0, min=0.0, label="Min. alan (px)"),
    ]

    def run(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        labels = inputs["labels"]
        min_area = float(params.get("min_area", 0.0))
        max_label = int(labels.max()) if labels.size else 0

        measurements: list[dict[str, Any]] = []
        for label in range(1, max_label + 1):
            mask = (labels == label).astype(np.uint8)
            area = float(cv2.countNonZero(mask))
            if area < min_area:
                continue

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            perimeter = float(cv2.arcLength(contour, closed=True))
            x, y, w, h = cv2.boundingRect(contour)

            moments = cv2.moments(mask, binaryImage=True)
            cx = moments["m10"] / moments["m00"] if moments["m00"] else float(x + w / 2)
            cy = moments["m01"] / moments["m00"] if moments["m00"] else float(y + h / 2)
            circularity = (4 * np.pi * area / (perimeter**2)) if perimeter > 0 else 0.0

            measurements.append(
                {
                    "label": label,
                    "area": area,
                    "perimeter": perimeter,
                    "centroid_x": cx,
                    "centroid_y": cy,
                    "bbox_x": x,
                    "bbox_y": y,
                    "bbox_w": w,
                    "bbox_h": h,
                    "circularity": circularity,
                }
            )
        return {"measurements": measurements}
