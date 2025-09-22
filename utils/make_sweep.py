# tools/make_sweep.py
import os, json, time, math, warnings, logging, sys
from copy import deepcopy
import gymnasium as gym, gymnasium_robotics
import mujoco

logging.getLogger("mujoco").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*Overriding environment.*already in registry.*")

# repo utils
BASE_DIR = "/home/ul/ul_student/ul_cep22/my_folders/drl_projekt"
sys.path.append(BASE_DIR)
from utils.change_parameters import get_geom_info  # noqa: E402

# ---------------- equal-volume helpers ----------------
def vol_box(hx, hy, hz): return 8.0 * hx * hy * hz
def _curved(shape): return shape in {"sphere", "ellipsoid"}

def sizes_equal_volume_from_box(hx, hy, hz, shape):
    V = vol_box(hx, hy, hz)
    if shape == "box":
        return [hx, hy, hz]
    elif shape == "ellipsoid":
        s = (V / ((4.0/3.0) * math.pi * hx * hy * hz)) ** (1.0/3.0)
        return [s*hx, s*hy, s*hz]           # Rx,Ry,Rz
    elif shape == "sphere":
        r = ((3.0 * V) / (4.0 * math.pi)) ** (1.0/3.0)
        return [r, 0.0, 0.0]                # [radius, 0, 0]
    elif shape == "cylinder":
        L = 2.0 * hz                        # preserve length along z
        r = (V / (math.pi * L)) ** 0.5
        return [r, hz, 0.0]                 # [radius, half_length, 0]
    else:
        raise ValueError(f"Unsupported shape: {shape}")

def get_baseline_type_and_size(env, geom_name="object0"):
    info = get_geom_info(env, geom_name)
    return info["type"], info["size"].tolist()

def get_baseline_mass(env, geom_name="object0"):
    m = env.unwrapped.model
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    bid = int(m.geom_bodyid[gid])
    return float(m.body_mass[bid])

# ---------------- job sanitizer (physics-aware) ----------------
def _sanitize_job(cfg, baseline_shape: str):
    """
    Enforce physically valid friction for curved shapes:
    - If the final object is curved (either baseline is curved and no override_object,
      or override_object.type is curved), then:
        * ensure condim >= 3
        * ensure spin/roll > 0 with minima (0.10, 0.05)
    """
    spin_min, roll_min = 0.10, 0.05

    # figure out which shape will be used in the run
    final_shape = baseline_shape
    if "override_object" in cfg and cfg["override_object"]:
        final_shape = cfg["override_object"]["type"]

    # nothing to do if not curved
    if not _curved(final_shape):
        return cfg

    # curved: sanitize explicit friction override if present
    fovr = cfg.get("override_friction")
    if fovr:
        fr = list(map(float, fovr.get("friction", [0.8, 0.0, 0.0])))
        # bump spin/roll if zero/small
        if fr[1] <= 0.0: fr[1] = spin_min
        if fr[2] <= 0.0: fr[2] = roll_min
        fovr["friction"] = fr
        # condim must be 3 for curved
        fovr["condim"] = max(int(fovr.get("condim", 1)), 3)
        cfg["override_friction"] = fovr
    # if no explicit friction override, runtime auto-policy in apply_override() will enforce it

    return cfg

# ---------------- sweep generation ----------------
def main():
    base_cfg_path = os.path.join(BASE_DIR, "configs", "test.json")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(BASE_DIR, "sweeps", stamp)
    os.makedirs(out_dir, exist_ok=True)

    with open(base_cfg_path, "r") as f:
        base = json.load(f)

    # === WHAT TO SWEEP ===
    AGENTS = ["ddpg", "td3", "sac"]

    # Envs: by default just the env in your base config.
    # Add others like: ENVS = ["FetchPush-v4", "FetchPickAndPlace-v4"]
    ENVS = [base["env_name"]]

    # Shape and friction grids
    SHAPES    = ["box", "ellipsoid", "sphere", "cylinder"]
    # slide, spin, roll  (spin/roll will be sanitized to nonzero for curved shapes)
    FRICTIONS = [[0.4, 0.0, 0.0], [0.8, 0.0, 0.0], [1.2, 0.0, 0.0]]
    # You can keep [1,3]; sanitizer will force 3 for curved final shapes
    CONDIMS   = [1, 3]

    jobs = []

    gym.register_envs(gymnasium_robotics)

    for env_name in ENVS:
        # Probe geometry for THIS env (so sizes match that env’s object0)
        env = gym.make(env_name)
        env.reset()
        try:
            baseline_shape, sz = get_baseline_type_and_size(env, "object0")
        finally:
            env.close()

        baseline_mass = base.get("override_object", {}).get("mass", None)
        if baseline_mass is None:
            # reopen briefly to read mass (cheap)
            env = gym.make(env_name)
            env.reset()
            try:
                baseline_mass = get_baseline_mass(env, "object0")
            finally:
                env.close()

        base_run = base.get("run_name", "run")

        for agent in AGENTS:
            # ---------- 0) Baseline ----------
            cfg = deepcopy(base)
            cfg["env_name"] = env_name
            cfg["agent"]    = agent
            cfg["run_name"] = f"{base_run}__{env_name}__{agent}__baseline"
            cfg.pop("override_object", None)
            cfg.pop("override_friction", None)
            jobs.append(_sanitize_job(cfg, baseline_shape))

            # ---------- 1) Shape sweep (no explicit friction override) ----------
            if baseline_shape == "box":
                hx, hy, hz = map(float, sz[:3])
                for shp in SHAPES:
                    sizes = sizes_equal_volume_from_box(hx, hy, hz, shp)
                    cfg = deepcopy(base)
                    cfg["env_name"] = env_name
                    cfg["agent"]    = agent
                    cfg["run_name"] = f"{base_run}__{env_name}__{agent}__shape_{shp}"
                    cfg["override_object"] = {
                        "name": "object0",
                        "type": shp,
                        "size": [float(sizes[0]), float(sizes[1]), float(sizes[2])],
                        "mass": float(baseline_mass),
                        "update_inertia": True
                    }
                    cfg.pop("override_friction", None)  # keep friction baseline
                    jobs.append(_sanitize_job(cfg, baseline_shape))
            else:
                # If baseline isn't a box, just echo that shape once
                cfg = deepcopy(base)
                cfg["env_name"] = env_name
                cfg["agent"]    = agent
                cfg["run_name"] = f"{base_run}__{env_name}__{agent}__shape_{baseline_shape}"
                cfg["override_object"] = {
                    "name": "object0",
                    "type": baseline_shape,
                    "size": [float(sz[0]), float(sz[1]), float(sz[2])],
                    "mass": float(baseline_mass),
                    "update_inertia": True
                }
                cfg.pop("override_friction", None)
                jobs.append(_sanitize_job(cfg, baseline_shape))

            # ---------- 3) Full factorial: shape × friction × condim ----------
            if baseline_shape == "box":
                hx, hy, hz = map(float, sz[:3])
                for shp in SHAPES:
                    sizes = sizes_equal_volume_from_box(hx, hy, hz, shp)
                    for fv in FRICTIONS:
                        for cd in CONDIMS:
                            # skip meaningless combos: curved + cd=1 will be sanitized to 3 → duplicate
                            if _curved(shp) and cd == 1:
                                continue

                            cfg = deepcopy(base)
                            cfg["env_name"] = env_name
                            cfg["agent"]    = agent
                            tag = f"{shp}__mu{fv[0]:.2f}_cd{cd}"
                            cfg["run_name"] = f"{base_run}__{env_name}__{agent}__shapeXfric_{tag}"
                            cfg["override_object"] = {
                                "name": "object0",
                                "type": shp,
                                "size": [float(sizes[0]), float(sizes[1]), float(sizes[2])],
                                "mass": float(baseline_mass),
                                "update_inertia": True
                            }
                            cfg["override_friction"] = {"name": "object0", "friction": fv, "condim": cd}
                            jobs.append(_sanitize_job(cfg, baseline_shape))

    # Write configs
    for i, c in enumerate(jobs):
        with open(os.path.join(out_dir, f"cfg_{i:03d}.json"), "w") as f:
            json.dump(c, f, indent=2, sort_keys=True)

    # Make/update stable symlink sweeps/latest_v2 -> out_dir
    latest_v2 = os.path.join(BASE_DIR, "sweeps", "latest_v2")
    try:
        if os.path.islink(latest_v2) or os.path.exists(latest_v2):
            os.remove(latest_v2)
        os.symlink(out_dir, latest_v2)
    except OSError:
        pass

    print(f"Wrote {len(jobs)} configs to: {out_dir}")
    print("Example:")
    print(f"  CFG_PATH={out_dir}/cfg_000.json python {BASE_DIR}/main_pars.py")
    print("Newest dir symlink:")
    print(f"  sweeps/latest_v2 -> {out_dir}")

if __name__ == "__main__":
    main()
