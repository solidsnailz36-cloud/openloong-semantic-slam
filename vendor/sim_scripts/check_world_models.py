#!/usr/bin/env python3
"""Which `model://` URIs does house.world ask for, and which are missing locally?

Gazebo falls back to the ONLINE model database (models.gazebosim.org) whenever a
`model://` URI cannot be resolved from GAZEBO_MODEL_PATH / ~/.gazebo/models.
On a container without egress that fetch blocks the whole world load and the
sim never starts stepping (sim_time stays 0) -- so every missing model is a
potential hang, not just a cosmetic warning.
"""
import os
import re
import sys

WORLD = sys.argv[1]
MODELS_DIR = sys.argv[2]

txt = open(WORLD, encoding="utf-8", errors="replace").read()

# every <uri>model://name</uri>
uris = re.findall(r"<uri>\s*model://([^<\s/]+)", txt)

# models that the world defines inline never need a database lookup
inline = set(re.findall(r"<model\s+name=['\"]([^'\"]+)['\"]", txt))
# ...but those names are world-instance names, not model:// names.  Keep them
# only as a hint for the report.

local = set()
if os.path.isdir(MODELS_DIR):
    for n in os.listdir(MODELS_DIR):
        if os.path.isdir(os.path.join(MODELS_DIR, n)):
            local.add(n)

want = sorted(set(uris))
missing = [u for u in want if u not in local]

print("world file      : %s  (%d bytes)" % (WORLD, len(txt)))
print("models dir      : %s  (%d dirs)" % (MODELS_DIR, len(local)))
print("model:// URIs   : %d  (%d distinct)" % (len(uris), len(want)))
print()
print("distinct model:// URIs asked for:")
for u in want:
    print("   %-52s %s" % (u, "local" if u in local else "** MISSING **"))
print()
print("MISSING (%d):" % len(missing))
for u in missing:
    print("   " + u)
print()
unused = sorted(local - set(want) - {"house.world"})
print("local but never referenced by an <uri> (%d): %s"
      % (len(unused), ", ".join(unused) if unused else "-"))
