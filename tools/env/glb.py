"""Minimal GLB (glTF 2.0 binary) writer / reader built on numpy.

Works with system python3 and with Blender's bundled python.
Conventions: +Y up, metres, UV origin top-left (glTF), COLOR_0 = normalized uint8 RGBA (linear).
"""
import json
import struct
import numpy as np

FLOAT = 5126
UBYTE = 5121
USHORT = 5123
UINT = 5125
ARRAY_BUFFER = 34962
ELEMENT_ARRAY_BUFFER = 34963

_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
_DTYPE = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}


class Part:
    """One primitive: a triangle list sharing one material."""

    def __init__(self, material, positions, normals, uvs=None, colors=None, indices=None):
        self.material = material
        self.positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
        self.normals = np.asarray(normals, dtype=np.float32).reshape(-1, 3)
        n = len(self.positions)
        self.uvs = None if uvs is None else np.asarray(uvs, dtype=np.float32).reshape(-1, 2)
        if colors is None:
            self.colors = None
        else:
            c = np.asarray(colors, dtype=np.float32).reshape(n, -1)
            if c.shape[1] == 1:
                c = np.repeat(c, 3, axis=1)
            if c.shape[1] == 3:
                c = np.concatenate([c, np.ones((n, 1), dtype=np.float32)], axis=1)
            self.colors = c
        if indices is None:
            indices = np.arange(n, dtype=np.uint32)
        self.indices = np.asarray(indices, dtype=np.uint32).reshape(-1)
        assert len(self.normals) == n, "normals/positions mismatch"
        assert self.uvs is None or len(self.uvs) == n, "uvs/positions mismatch"
        assert len(self.indices) % 3 == 0
        assert self.indices.size == 0 or int(self.indices.max()) < n

    @property
    def triangles(self):
        return len(self.indices) // 3


def material_def(name, color=(0.8, 0.8, 0.8, 1.0), metallic=0.0, roughness=0.8, emissive=None,
                 alpha_mode="OPAQUE", alpha_cutoff=0.5, double_sided=False):
    c = list(color) + [1.0] * (4 - len(color))
    m = {
        "name": name,
        "pbrMetallicRoughness": {
            "baseColorFactor": [float(x) for x in c[:4]],
            "metallicFactor": float(metallic),
            "roughnessFactor": float(roughness),
        },
    }
    if emissive is not None:
        m["emissiveFactor"] = [float(x) for x in emissive]
    if alpha_mode != "OPAQUE":
        m["alphaMode"] = alpha_mode
        if alpha_mode == "MASK":
            m["alphaCutoff"] = float(alpha_cutoff)
    if double_sided:
        m["doubleSided"] = True
    return m


def write_glb(path, nodes, materials, generator="apex-velocity env tools", extras=None):
    """nodes: list of dict(name=str, parts=[Part], translation=(x,y,z)?)
    materials: dict name -> material_def(...) (only the ones referenced are written)."""
    bin_chunks = []
    offset = 0
    buffer_views = []
    accessors = []

    def add_view(data_bytes, target):
        nonlocal offset
        pad = (-offset) % 4
        if pad:
            bin_chunks.append(b"\x00" * pad)
            offset += pad
        buffer_views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(data_bytes), "target": target})
        bin_chunks.append(data_bytes)
        offset += len(data_bytes)
        return len(buffer_views) - 1

    def add_accessor(arr, comp_type, type_name, target, normalized=False, minmax=False):
        view = add_view(arr.tobytes(), target)
        acc = {"bufferView": view, "componentType": comp_type, "count": int(arr.shape[0]), "type": type_name}
        if normalized:
            acc["normalized"] = True
        if minmax:
            acc["min"] = [float(x) for x in arr.min(axis=0)]
            acc["max"] = [float(x) for x in arr.max(axis=0)]
        accessors.append(acc)
        return len(accessors) - 1

    used = []
    for node in nodes:
        for p in node["parts"]:
            if p.material not in used:
                used.append(p.material)
    mat_index = {name: i for i, name in enumerate(used)}
    gl_materials = [materials[name] for name in used]

    gl_meshes = []
    gl_nodes = []
    for node in nodes:
        prims = []
        for p in node["parts"]:
            if p.triangles == 0:
                continue
            attrs = {
                "POSITION": add_accessor(np.ascontiguousarray(p.positions, dtype=np.float32), FLOAT, "VEC3", ARRAY_BUFFER, minmax=True),
                "NORMAL": add_accessor(np.ascontiguousarray(p.normals, dtype=np.float32), FLOAT, "VEC3", ARRAY_BUFFER),
            }
            if p.uvs is not None:
                attrs["TEXCOORD_0"] = add_accessor(np.ascontiguousarray(p.uvs, dtype=np.float32), FLOAT, "VEC2", ARRAY_BUFFER)
            if p.colors is not None:
                c8 = np.clip(np.round(p.colors * 255.0), 0, 255).astype(np.uint8)
                attrs["COLOR_0"] = add_accessor(np.ascontiguousarray(c8), UBYTE, "VEC4", ARRAY_BUFFER, normalized=True)
            nverts = len(p.positions)
            if nverts <= 65535:
                idx = add_accessor(p.indices.astype(np.uint16), USHORT, "SCALAR", ELEMENT_ARRAY_BUFFER)
            else:
                idx = add_accessor(p.indices.astype(np.uint32), UINT, "SCALAR", ELEMENT_ARRAY_BUFFER)
            prims.append({"attributes": attrs, "indices": idx, "material": mat_index[p.material], "mode": 4})
        gl_meshes.append({"name": node["name"], "primitives": prims})
        gn = {"name": node["name"], "mesh": len(gl_meshes) - 1}
        if node.get("translation") is not None:
            gn["translation"] = [float(x) for x in node["translation"]]
        gl_nodes.append(gn)

    pad = (-offset) % 4
    if pad:
        bin_chunks.append(b"\x00" * pad)
        offset += pad
    gltf = {
        "asset": {"version": "2.0", "generator": generator},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(gl_nodes)))}],
        "nodes": gl_nodes,
        "meshes": gl_meshes,
        "materials": gl_materials,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": offset}],
    }
    if extras:
        gltf["asset"]["extras"] = extras
    js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    js += b" " * ((-len(js)) % 4)
    bin_data = b"".join(bin_chunks)
    total = 12 + 8 + len(js) + 8 + len(bin_data)
    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, total))
        f.write(struct.pack("<I4s", len(js), b"JSON"))
        f.write(js)
        f.write(struct.pack("<I4s", len(bin_data), b"BIN\x00"))
        f.write(bin_data)
    return total


def read_glb(path):
    """Returns (gltf_json, accessor_reader) where accessor_reader(i) -> numpy array (float for normalized)."""
    with open(path, "rb") as f:
        data = f.read()
    magic, version, total = struct.unpack_from("<4sII", data, 0)
    assert magic == b"glTF" and version == 2, "not a GLB v2"
    pos = 12
    js = None
    bin_data = b""
    while pos < len(data):
        clen, ctype = struct.unpack_from("<I4s", data, pos)
        pos += 8
        chunk = data[pos:pos + clen]
        pos += clen
        if ctype == b"JSON":
            js = json.loads(chunk.decode("utf-8"))
        elif ctype == b"BIN\x00":
            bin_data = chunk

    def accessor(i, raw=False):
        acc = js["accessors"][i]
        bv = js["bufferViews"][acc["bufferView"]]
        dt = np.dtype(_DTYPE[acc["componentType"]])
        ncomp = _NCOMP[acc["type"]]
        start = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
        stride = bv.get("byteStride", 0)
        count = acc["count"]
        if stride and stride != dt.itemsize * ncomp:
            out = np.zeros((count, ncomp), dtype=dt)
            for k in range(count):
                out[k] = np.frombuffer(bin_data, dtype=dt, count=ncomp, offset=start + k * stride)
            arr = out
        else:
            arr = np.frombuffer(bin_data, dtype=dt, count=count * ncomp, offset=start).reshape(count, ncomp)
        if acc.get("normalized") and not raw:
            arr = arr.astype(np.float32) / float(np.iinfo(dt).max)
        return arr

    return js, accessor
