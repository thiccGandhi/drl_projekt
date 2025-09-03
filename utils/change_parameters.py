
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


# ---------- low-level helpers ----------

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
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    t_int = int(m.geom_type[gid])
    return {
        "gid": gid,
        "type": _type_str_from_int(t_int),
        "size": m.geom_size[gid].copy(),              # (3,)
        "dataid": int(m.geom_dataid[gid]),            # -1 unless mesh/hfield/etc.
        "body_id": int(m.geom_bodyid[gid]),
        "friction": m.geom_friction[gid].copy(),      # [slide, spin, roll]
        "condim": int(m.geom_condim[gid]),            # # of friction dims used
    }


def assert_geom(env, geom_name: str, expect_type: str, expect_size, atol: float = 1e-9) -> tuple[bool, dict]:
    """
    Check that live model has requested type and size (prefix compare on size length).
    """
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
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
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

    mujoco.mj_forward(m, d)
    return gid

# maybe we could use this a another parameter to change not sure if needed
# ---------- physics consistency: mass & diagonal inertia ----------

def set_body_mass_inertia_for_geom(env, geom_name: str, mass: float) -> tuple[int, np.ndarray]:
    """
    Update parent body's mass and *diagonal* inertia assuming a single solid primitive.
    Supported: ellipsoid, box, sphere. (Extend as needed.)
    Inertia is written in the body's frame and assumes the geom axes align with the body.
    """
    m, d = env.unwrapped.model, env.unwrapped.data
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    bid = int(m.geom_bodyid[gid])
    t = int(m.geom_type[gid]); sz = m.geom_size[gid]

    # Ellipsoid radii Rx,Ry,Rz
    if t == getattr(mujoco.mjtGeom, "mjGEOM_ELLIPSOID"):
        Rx, Ry, Rz = float(sz[0]), float(sz[1]), float(sz[2])
        Ixx = 0.2 * mass * (Ry * Ry + Rz * Rz)
        Iyy = 0.2 * mass * (Rx * Rx + Rz * Rz)
        Izz = 0.2 * mass * (Rx * Rx + Ry * Ry)

    # Box half-sizes hx,hy,hz  (full lengths are 2*hx etc.)
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
    mujoco.mj_forward(m, d)
    return bid, m.body_inertia[bid].copy()


# ---------- high-level convenience ----------

def apply_override(env, name: str, typ: str, size, *, mass: float | None = None,
                   update_inertia: bool = False, atol: float = 1e-9) -> dict:
    """
    One-shot: set shape, assert success, optionally update mass/inertia.
    Returns a dict with before/after info and flags.
    """
    before = get_geom_info(env, name)
    gid = set_geom_shape(env, name, typ, size)
    ok, after = assert_geom(env, name, typ, size, atol=atol)

    out = {
        "gid": gid,
        "ok": ok,
        "before": before,
        "after": after,
        "inertia_updated": False,
        "body_id": after["body_id"],
    }
    if update_inertia and (mass is not None):
        _, I = set_body_mass_inertia_for_geom(env, name, float(mass))
        out["inertia_updated"] = True
        out["body_inertia"] = I
        out["body_mass"] = float(mass)
    return out



# --- just visual tests ---


def snapshot_geom_and_body(env, geom_name: str) -> dict:
    """
    Return a structured snapshot of the geom and its parent body
    (what's effectively 'in the XML' after compile).
    """
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
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
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
    - or set any of slide/spin/roll individually.
    - condim: 1 (sliding only) or 3 (sliding + spin + roll), etc.
    Returns: geom id.
    """
    m, d = env.unwrapped.model, env.unwrapped.data
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, geom_name)

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
        m.geom_condim[gid] = int(condim)

    mujoco.mj_forward(m, d)
    return gid

def assert_friction(env, geom_name: str, expect, atol: float = 1e-9) -> tuple[bool, np.ndarray]:
    """Check the geom's friction matches expected (length-3)."""
    fr = get_geom_friction(env, geom_name)
    ok = np.allclose(fr, np.asarray(expect, float).reshape(3), atol=atol, rtol=0.0)
    return ok, fr

def contact_friction_product(env, geom_a: str, geom_b: str) -> np.ndarray:
    """
    Compute the effective contact friction (element-wise product) that would be used
    between two geoms, ignoring any pair-specific overrides.
    """
    fa = get_geom_friction(env, geom_a)
    fb = get_geom_friction(env, geom_b)
    return fa * fb

def apply_friction_override(env, name: str, friction, condim: int | None = None,
                            atol: float = 1e-9) -> dict:
    """
    One-shot friction override with assertion; returns before/after.
    """
    before = get_geom_info(env, name)
    set_geom_friction(env, name, friction=friction, condim=condim)
    ok, fr_after = assert_friction(env, name, friction, atol=atol)
    after = get_geom_info(env, name)
    return {
        "ok": ok,
        "before": {"friction": before["friction"], "condim": before["condim"]},
        "after":  {"friction": fr_after,           "condim": after["condim"]},
    }