"""Generates realistic synthetic results for demo/testing purposes."""
import random

def generate_demo_results():
    random.seed(42)
    def ts(sec):
        return f"{sec//3600:02d}:{(sec%3600)//60:02d}:{sec%60:02d}"

    violations = {
        "helmet_less":   {"label":"Helmet-less Riding",   "total_count":23},
        "wrong_side":    {"label":"Wrong-side Driving",    "total_count":4},
        "signal_jump":   {"label":"Signal Jumping",        "total_count":7},
        "phone_use":     {"label":"Mobile Phone Use",      "total_count":11},
        "triple_riding": {"label":"Triple Riding",         "total_count":3},
    }
    counts = [23,4,7,11,3]
    for (k,v), n in zip(violations.items(), counts):
        secs = sorted(random.randint(10, 4000) for _ in range(n))
        v["instances"] = [{"ts": ts(s), "timestamp": float(s), "frame": s*30, "conf": round(random.uniform(0.55,0.92),2)} for s in secs]
        v["top3_annotations"] = [{"ts": v["instances"][i]["ts"], "frame": v["instances"][i]["frame"],
                                   "bbox":[100,150,300,400], "conf": v["instances"][i]["conf"]} for i in range(min(3,n))]

    junctions = [
        ("T_JUNCTION",75),("T_JUNCTION",210),("X_JUNCTION",430),("T_JUNCTION",620),
        ("ROUNDABOUT",890),("T_JUNCTION",1100),("X_JUNCTION",1380),("Y_JUNCTION",1600),
        ("T_JUNCTION",1900),("FLYOVER",2200),("T_JUNCTION",2500),("X_JUNCTION",2800),
        ("T_JUNCTION",3100),("T_JUNCTION",3400),("Y_JUNCTION",3700),("T_JUNCTION",4000),
    ]
    from collections import defaultdict
    type_counts = defaultdict(int)
    j_instances = []
    for jtype, sec in junctions:
        type_counts[jtype] += 1
        j_instances.append({"junction_type": jtype, "ts": ts(sec), "timestamp": float(sec), "frame": sec*30, "line_count": random.randint(8,20)})

    density = []
    for s in range(0, 4194, 2):
        density.append({"timestamp": float(s), "ts_str": ts(s), "count": random.randint(2,18)})

    return {
        "meta": {"video":"Bangalore_City_Drive_demo.mp4","fps":30,"total_frames":125817,
                 "duration_seconds":4193,"duration_hms":"01:09:53",
                 "resolution":"1280x720","frame_skip":15,"method":"DEMO (synthetic)"},
        "violations": violations,
        "junctions": {"total_count":len(junctions),"type_breakdown":dict(type_counts),"instances":j_instances},
        "vehicles": {"total_unique":847,"categories":{
            "2W":{"count":412,"pct":48.6},"LMV":{"count":338,"pct":39.9},
            "HMV":{"count":72,"pct":8.5},"OTHER":{"count":25,"pct":3.0}}},
        "density_timeline": density,
    }
