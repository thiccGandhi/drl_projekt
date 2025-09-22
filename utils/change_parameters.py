# Utilities to override a MuJoCo geom's shape and friction in a live Gymnasium env.

from __future__ import annotations
import json
import numpy as np
import mujoco

# ---------- JSON printing ----------
def _to_serializable(x):
    try:
        import numpy as _np
        if isinstance(x, _np.ndarray):
            return x.tolist()
    except Exception:
        pass
    return x

def jprint(tag: str, obj: dict):
    """Stable JSON print for logs (handles numpy arrays)."""
    print(tag, json.dumps(obj, indent=2, default=_to_serializable))

# ---------- helpers for (de)numpy ----------
def _py(o):
    import numpy as _np
    if isinstance(o, _np.ndarray):          return o.tolist()
    if isinstance(o, (_np.floating,)):      return float(o)
    if isinstance(o, (_np.integer,)):       return int(o)
    if isinstance(o, (_np.bool_,)):         return bool(o)
    return o

def _deep_py(o):
    if isinstance(o, dict):          return {k: _deep_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_deep_py(v) for v in o]
    return _py(o)

def _mj_setConst(m, d):
    """Version-safe wrapper for mujoco.mj_setConst."""
    try:
        mujoco.mj_setConst(m, d)  # newer Python bindings
    except TypeError:
        mujoco.mj_setConst(m)     # older bindings

# ---------- low-level helpers ----------
def _name2geomid(m, name: str) -> int:
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
    if gid < 0:
        raise ValueError(f"Geom '{name}' not found in model.")
    return gid

def _type_enum(type_str: str) -> int:
    """Return the mjGEOM_* enum for a lowercase type string, e.g. 'box' -> mjGEOM_BOX."""
    enum_name = f"mjGEOM_{type_str.upper()}"
    try:
        return getattr(mujoco.mjtGeom, enum_name)
    except AttributeError as e:
        raise ValueError(f"Unknown geom type '{type_str}'.") from e

def _type_str_from_int(t_int: int) -> str:
    """Inverse of _type_enum: int -> lowercase type name."""
    for name in dir(mujoco.mjtGeom):
        if name.startswith("mjGEOM_") and getattr(mujoco.mjtGeom, name) == t_int:
            return name.replace("mjGEOM_", "").lower()
    return f"unknown({t_int})"

# ---------- read/inspect ----------
def get_geom_info(env, geom_name: str) -> dict:
    """
    Inspect key fields of a geom in the live mjModel/mjData.
    Returns: {'gid','type','size','dataid','body_id','friction','condim'}
    """
    m, d = env.unwrapped.model, env.unwrapped.data
    gid = _name2geomid(m, geom_name)
    t_int = int(m.geom_type[gid])
    return {
        "gid": gid,
        "type": _type_str_from_int(t_int),
        "size": m.geom_size[gid].copy(),         # (3,)
        "dataid": int(m.geom_dataid[gid]),       # -1 unless mesh/hfield/etc.
        "body_id": int(m.geom_bodyid[gid]),
        "friction": m.geom_friction[gid].copy(), # [slide, spin, roll]
        "condim": int(m.geom_condim[gid]),       # # of friction dims used
    }

def assert_geom(env, geom_name: str, expect_type: str, expect_size, atol: float = 1e-9) -> tuple[bool, dict]:
    """Check that live model has requested type and size (prefix compare on size length)."""
    info = get_geom_info(env, geom_name)
    ok_type = (info["type"] == expect_type.lower())
    sz = np.asarray(expect_size, float)
    ok_size = np.allclose(info["size"][: len(sz)], sz, atol=atol, rtol=0.0)
    return (ok_type and ok_size), info

# ---------- write/mutate ----------
def set_geom_shape(env, geom_name: str, new_type: str, new_size) -> int:
    """
    Mutate a geom's collision/visual type and size at runtime; calls mj_forward().
    NOTE: This does NOT recompute mass/inertia of the parent body.
    Returns: geom id (int).
    """
    m, d = env.unwrapped.model, env.unwrapped.data
    gid = _name2geomid(m, geom_name)
    m.geom_type[gid] = _type_enum(new_type)

    sz = np.zeros(3, dtype=np.float64)
    ns = np.asarray(new_size, dtype=float)
    if ns.size > 3:
        raise ValueError("new_size must have length ≤ 3.")
    sz[: ns.size] = ns
    m.geom_size[gid] = sz

    # If not a mesh, ensure no stray mesh data id is attached
    if new_type.lower() != "mesh":
        m.geom_dataid[gid] = -1

    _mj_setConst(m, d)
    mujoco.mj_forward(m, d)
    return gid

# ---------- physics consistency: mass & diagonal inertia ----------
def set_body_mass_inertia_for_geom(env, geom_name: str, mass: float) -> tuple[int, np.ndarray]:
    """
    Update parent body's mass and *diagonal* inertia assuming a single solid primitive.
    Supported: ellipsoid, box, sphere.
    """
    m, d = env.unwrapped.model, env.unwrapped.data
    gid = _name2geomid(m, geom_name)
    bid = int(m.geom_bodyid[gid])
    t = int(m.geom_type[gid]); sz = m.geom_size[gid]

    # Ellipsoid radii Rx,Ry,Rz
    if t == getattr(mujoco.mjtGeom, "mjGEOM_ELLIPSOID"):
        Rx, Ry, Rz = float(sz[0]), float(sz[1]), float(sz[2])
        Ixx = 0.2 * mass * (Ry * Ry + Rz * Rz)
        Iyy = 0.2 * mass * (Rx * Rx + Rz * Rz)
        Izz = 0.2 * mass * (Rx * Rx + Ry * Ry)

    # Box half-sizes hx,hy,hz
    elif t == getattr(mujoco.mjtGeom, "mjGEOM_BOX"):
        hx, hy, hz = float(sz[0]), float(sz[1]), float(sz[2])
        Ixx = (1.0 / 3.0) * mass * (hy * hy + hz * hz)
        Iyy = (1.0 / 3.0) * mass * (hx * hx + hz * hz)
        Izz = (1.0 / 3.0) * mass * (hx * hx + hy * hy)

    # Sphere radius r
    elif t == getattr(mujoco.mjtGeom, "mjGEOM_SPHERE"):
        r = float(sz[0])
        Ixx = Iyy = Izz = 0.4 * mass * r * r

    else:
        # Leave previous inertia for unsupported shapes (capsule/cylinder/mesh etc.)
        m.body_mass[bid] = mass
        mujoco.mj_forward(m, d)
        return bid, m.body_inertia[bid].copy()

    m.body_mass[bid] = mass
    m.body_inertia[bid] = np.array([Ixx, Iyy, Izz], dtype=np.float64)
    _mj_setConst(m, d)
    mujoco.mj_forward(m, d)
    return bid, m.body_inertia[bid].copy()

# ---------- high-level convenience ----------
def apply_override(env, name: str, typ: str, size, *, mass: float | None = None,
                   update_inertia: bool = False, atol: float = 1e-9,
                   auto_adjust_friction_for_curved: bool = True,
                   spin_min: float = 0.10, roll_min: float = 0.05) -> dict:
    before = get_geom_info(env, name)
    gid = set_geom_shape(env, name, typ, size)
    ok, after = assert_geom(env, name, typ, size, atol=atol)

    out = {
        "gid": gid, "ok": ok,
        "before": before, "after": after,
        "inertia_updated": False, "body_id": after["body_id"],
    }

    # If user asked to update inertia but didn't pass mass, keep current body mass.
    if update_inertia and (mass is None):
        m = env.unwrapped.model
        bid = after["body_id"]
        mass = float(m.body_mass[bid])

    if update_inertia and (mass is not None):
        _, I = set_body_mass_inertia_for_geom(env, name, float(mass))
        out["inertia_updated"] = True
        out["body_inertia"] = I
        out["body_mass"] = float(mass)

    # Auto friction policy for curved shapes (unless overridden explicitly elsewhere)
    if auto_adjust_friction_for_curved and after["type"] in ("ellipsoid", "sphere"):
        # use the CURRENT friction vector post-shape-change
        m = env.unwrapped.model
        fr = m.geom_friction[gid].copy()
        fr[1] = max(fr[1], float(spin_min))   # spin
        fr[2] = max(fr[2], float(roll_min))   # roll
        set_geom_friction(env, name, friction=fr, condim=3)

    return out

def apply_overrides_from_config(env, cfg: dict, geom_name: str = "object0",
                                reseed_after: bool = True, seed: int | None = None) -> dict:
    """
    Apply shape + optional inertia + friction overrides from a config dict.
    Returns {"before":..., "after":...}.
    """
    # Snapshot before
    snap_before = snapshot_geom_and_body(env, geom_name)

    # 1) Shape/inertia
    ovr = cfg.get("override_object")
    if ovr:
        res_shape = apply_override(
            env,
            name=ovr.get("name", geom_name),
            typ=ovr["type"],
            size=ovr["size"],
            mass=ovr.get("mass", None),
            update_inertia=bool(ovr.get("update_inertia", False)),
            auto_adjust_friction_for_curved=True,
        )
        cfg["object_inertia_updated"] = res_shape["inertia_updated"]

    # 2) Explicit friction (overrides any auto policy)
    fovr = cfg.get("override_friction")
    if fovr:
        apply_friction_override(
            env,
            name=fovr.get("name", geom_name),
            friction=fovr["friction"],
            condim=fovr.get("condim", None),
            atol=1e-9,
        )

    # Reseed so placements respect new geometry
    if reseed_after:
        try:
            env.reset(seed=seed)
        except Exception:
            pass

    # Snapshot after
    snap_after = snapshot_geom_and_body(env, geom_name)

    # Mirror fields for logging
    cfg["object_gid"]           = _py(snap_before["geom"]["gid"])
    cfg["object_type_initial"]  = snap_before["geom"]["type"]
    cfg["object_size_initial"]  = _py(snap_before["geom"]["size"])
    cfg["body_mass_initial"]    = _py(snap_before["body"]["mass"])
    cfg["body_inertia_initial"] = _py(snap_before["body"]["inertia_diag"])
    cfg["friction_initial"]     = _py(snap_before["geom"]["friction"])
    cfg["condim_initial"]       = _py(snap_before["geom"]["condim"])

    cfg["object_type_final"]    = snap_after["geom"]["type"]
    cfg["object_size_final"]    = _py(snap_after["geom"]["size"])
    cfg["body_mass_final"]      = _py(snap_after["body"]["mass"])
    cfg["body_inertia_final"]   = _py(snap_after["body"]["inertia_diag"])
    cfg["friction_final"]       = _py(snap_after["geom"]["friction"])
    cfg["condim_final"]         = _py(snap_after["geom"]["condim"])

    return {"before": snap_before, "after": snap_after}

# --- visual snapshot helper ---
def snapshot_geom_and_body(env, geom_name: str) -> dict:
    m, d = env.unwrapped.model, env.unwrapped.data
    info = get_geom_info(env, geom_name)
    bid = int(m.geom_bodyid[info["gid"]])
    snap = {
        "geom": {
            "name": geom_name,
            "gid": info["gid"],
            "type": info["type"],
            "size": info["size"].copy(),
            "dataid": info["dataid"],
            "friction": m.geom_friction[info["gid"]].copy(),
            "condim": int(m.geom_condim[info["gid"]]),
        },
        "body": {
            "bid": bid,
            "mass": float(m.body_mass[bid]),
            "inertia_diag": m.body_inertia[bid].copy(),
        },
    }
    return snap

# ---------- friction utils ----------
def get_geom_friction(env, geom_name: str) -> np.ndarray:
    """Return geom's friction vector [slide, spin, roll]."""
    m = env.unwrapped.model
    gid = _name2geomid(m, geom_name)
    return m.geom_friction[gid].copy()

def set_geom_friction(env,
                      geom_name: str,
                      friction: np.ndarray | list | tuple | None = None,
                      *,
                      slide: float | None = None,
                      spin:  float | None = None,
                      roll:  float | None = None,
                      condim: int | None = None) -> int:
    """
    Set a geom's friction coefficients and optionally its condim.
    - friction: iterable of length 3 -> [slide, spin, roll]
    - condim: 1 (sliding only) or 3 (sliding + spin + roll)
    Returns: geom id.
    """
    m, d = env.unwrapped.model, env.unwrapped.data
    gid = _name2geomid(m, geom_name)

    vec = m.geom_friction[gid].copy()
    if friction is not None:
        fr = np.asarray(friction, dtype=float).reshape(3)
        vec[:] = fr
    if slide is not None: vec[0] = float(slide)
    if spin  is not None: vec[1] = float(spin)
    if roll  is not None: vec[2] = float(roll)

    # non-negativity guard
    vec = np.maximum(vec, 0.0)
    m.geom_friction[gid] = vec

    if condim is not None:
        # normalize to {1,3} to avoid accidental 0/2/etc.
        c = int(condim)
        if c <= 1: c = 1
        elif c >= 3: c = 3
        m.geom_condim[gid] = c

    _mj_setConst(m, d)
    mujoco.mj_forward(m, d)
    return gid

def assert_friction(env, geom_name: str, expect, atol: float = 1e-9) -> tuple[bool, np.ndarray]:
    """Check the geom's friction matches expected (length-3)."""
    fr = get_geom_friction(env, geom_name)
    ok = np.allclose(fr, np.asarray(expect, float).reshape(3), atol=atol, rtol=0.0)
    return ok, fr

def contact_friction_product(env, geom_a: str, geom_b: str) -> np.ndarray:
    """Element-wise product of the two geoms' friction vectors."""
    fa = get_geom_friction(env, geom_a)
    fb = get_geom_friction(env, geom_b)
    return fa * fb

def apply_friction_override(env, name: str, friction, condim: int | None = None,
                            atol: float = 1e-9) -> dict:
    """One-shot friction override with assertion; returns before/after."""
    before = get_geom_info(env, name)
    set_geom_friction(env, name, friction=friction, condim=condim)
    ok, fr_after = assert_friction(env, name, friction, atol=atol)
    after = get_geom_info(env, name)
    return {
        "ok": ok,
        "before": {"friction": _py(before["friction"]), "condim": _py(before["condim"])},
        "after":  {"friction": _py(fr_after),           "condim": _py(after["condim"])},
    }
