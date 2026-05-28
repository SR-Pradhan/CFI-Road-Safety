"""Exports results to JSON and prints summary."""
import json, os

class ResultExporter:
    def __init__(self, output_dir):
        self.out = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def export_json(self, results: dict) -> str:
        path = os.path.join(self.out, "results.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        return path

    def print_summary(self, results: dict):
        v = results.get("violations", {})
        j = results.get("junctions", {})
        vh = results.get("vehicles", {})
        print("\n" + "="*50)
        print("RESULTS SUMMARY")
        print("="*50)
        total_v = sum(x.get("total_count",0) for x in v.values())
        print(f"Total violations : {total_v}")
        for key, data in v.items():
            if data.get("total_count", 0):
                print(f"  {data['label']:28s}: {data['total_count']}")
        print(f"Total junctions  : {j.get('total_count', 0)}")
        for k, cnt in j.get("type_breakdown", {}).items():
            print(f"  {k:20s}: {cnt}")
        print(f"Total vehicles   : {vh.get('total_unique', 0)}")
        for cat, d in vh.get("categories", {}).items():
            print(f"  {cat:8s}: {d['count']} ({d['pct']}%)")
        print("="*50)
