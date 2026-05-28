"""Vehicle classification summarizer for tracker output."""
from collections import defaultdict

COCO_TO_CATEGORY = {
    "motorcycle": "2W", "bicycle": "2W",
    "car": "LMV", "truck": "HMV", "bus": "HMV",
}

class VehicleClassifier:
    def summarize(self, all_tracks: dict) -> dict:
        cat_counts = defaultdict(int)
        for track in all_tracks.values():
            cat = track.get("category", "OTHER")
            cat_counts[cat] += 1
        total = max(sum(cat_counts.values()), 1)
        return {
            "total_unique": sum(cat_counts.values()),
            "categories": {
                cat: {"count": cnt, "pct": round(cnt/total*100, 1)}
                for cat, cnt in cat_counts.items()
            }
        }
