"""Frozen scene routing and review-backed field monitoring."""
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from uuid import uuid4

from common import CLASSES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = PROJECT_ROOT / 'outputs' / 'train_improve_late'
MONITOR_PATH = EXPERIMENT / 'routing_monitor' / 'events.sqlite3'
# Use the original paired seed, independent of held-out seed rankings.
FROZEN_CHECKPOINTS = {
    'A': (EXPERIMENT / 'bg_replace' / 'baseline_s42' / 'best.pt',
          'c5635de07769c046630d3d647c6bd5a8ffff5d89a1eef07bfcb362b4521654ed'),
    'C': (EXPERIMENT / 'pad_aspect' / 'bg_replace_pad_s42' / 'best.pt',
          '66c0164a1c36228646352497213634ed9fc4f9ec640c4b5f27a930a8ee2b2a70'),
}


@dataclass(frozen=True)
class Route:
    scene: str
    scene_missing: bool
    variant: str


def route_scene(scene):
    if scene is None or scene == '':
        return Route('field', True, 'C')
    if scene == 'field':
        return Route('field', False, 'C')
    if scene == 'greenhouse':
        return Route('greenhouse', False, 'A')
    raise ValueError('Capture scene must be field, greenhouse, or empty.')


def displayed_size(image):
    """Return width and height in the displayed EXIF orientation."""
    if image.getexif().get(274) in (5, 6, 7, 8):
        return image.height, image.width
    return image.size


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


class ReviewStore:
    """Persist one prediction and optional confirmed label per prediction ID."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as db, db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id TEXT PRIMARY KEY,
                    predicted_at TEXT NOT NULL,
                    scene TEXT NOT NULL,
                    scene_missing INTEGER NOT NULL,
                    variant TEXT NOT NULL,
                    predicted_label TEXT NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    reviewed_label TEXT,
                    reviewed_at TEXT
                )
            """)

    def connect(self):
        return sqlite3.connect(self.path, timeout=15)

    def save_prediction(self, route, predicted_label, size):
        prediction_id = uuid4().hex
        width, height = size
        with closing(self.connect()) as db, db:
            db.execute("""
                INSERT INTO predictions
                    (id, predicted_at, scene, scene_missing, variant, predicted_label, width, height)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (prediction_id, utc_now(), route.scene, route.scene_missing, route.variant,
                  predicted_label, width, height))
        return prediction_id

    def save_review(self, prediction_id, label):
        if label not in CLASSES:
            raise ValueError('Select a confirmed label before saving a review.')
        with closing(self.connect()) as db, db:
            updated = db.execute("""
                UPDATE predictions SET reviewed_label = ?, reviewed_at = ?
                WHERE id = ? AND reviewed_label IS NULL
            """, (label, utc_now(), prediction_id))
            if updated.rowcount == 1:
                return
            existing = db.execute("""
                SELECT reviewed_label FROM predictions WHERE id = ?
            """, (prediction_id,)).fetchone()
            if existing is None:
                raise ValueError('Prediction ID was not found.')
            if existing[0] != label:
                raise ValueError('This prediction already has a different confirmed label.')

    def field_metrics(self):
        with closing(self.connect()) as db:
            total, reviewed, pending, late, missed, landscape, alarms = db.execute("""
                SELECT COUNT(*), COUNT(reviewed_label),
                    COUNT(CASE WHEN scene_missing = 1 AND reviewed_label IS NULL THEN 1 END),
                    COUNT(CASE WHEN reviewed_label = 'late_blight' THEN 1 END),
                    COUNT(CASE WHEN reviewed_label = 'late_blight'
                               AND predicted_label = 'healthy' THEN 1 END),
                    COUNT(CASE WHEN width > height AND reviewed_label = 'healthy' THEN 1 END),
                    COUNT(CASE WHEN width > height AND reviewed_label = 'healthy'
                               AND predicted_label != 'healthy' THEN 1 END)
                FROM predictions WHERE scene = 'field'
            """).fetchone()
        return {
            'field_predictions': total,
            'reviewed_field_predictions': reviewed,
            'pending_missing_scene_reviews': pending,
            'late_to_healthy': {
                'count': missed, 'reviewed_late_blight': late,
                'rate': missed / late if late else None,
            },
            'landscape_healthy_false_alarms': {
                'count': alarms, 'reviewed_landscape_healthy': landscape,
                'rate': alarms / landscape if landscape else None,
            },
        }
