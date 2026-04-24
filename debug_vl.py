from graph.graph import build_vl_segments
from validate_layer.validator import ValidationLayer

segments = build_vl_segments()
for s in segments:
    print(
        f"ID: {s.id}, Start: {s.start}, End: {s.end}, Limit: {s.limit_ms}, SH: {s.spatial_headway}"
    )

vl = ValidationLayer()
dtz = vl.compute_dtz(582.0)
print(f"DTZ at 582.0: {dtz}")
