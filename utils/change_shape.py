# change_shape.py
# Utilities to programmatically override a MuJoCo geom's shape in a live Gymnasium env.

from __future__ import annotations
import numpy as np
import mujoco


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
    Read key fields of a geom from the live mjModel/mjData.
    Returns: {'gid', 'type', 'size', 'dataid', 'body_id'}
    """
    m, d = env.unwrapped.model, env.unwrapped.data
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    t_int = int(m.geom_type[gid])
    return {
        "gid": gid,
        "type": _type_str_from_int(t_int),
        "size": m.geom_size[gid].copy(),          # (3,) array
        "dataid": int(m.geom_dataid[gid]),        # -1 unless mesh/hfield/etc.
        "body_id": int(m.geom_bodyid[gid]),
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


# ---------- (optional) physics consistency: mass & diagonal inertia ----------

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
