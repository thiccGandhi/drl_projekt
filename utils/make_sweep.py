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

# ---------------- sweep generation ----------------
def main():
    base_cfg_path = os.path.join(BASE_DIR, "configs", "test.json")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(BASE_DIR, "sweeps", stamp)
    os.makedirs(out_dir, exist_ok=True)

    base = json.load(open(base_cfg_path))

    # === WHAT TO SWEEP ===
    AGENTS = ["ddpg", "td3", "sac"]

    # Envs: by default just the env in your base config.
    # Add others like: ENVS = ["FetchPush-v4", "FetchPickAndPlace-v4"]
    ENVS = [base["env_name"]]

    # Shape and friction grids
    SHAPES    = ["box", "ellipsoid", "sphere", "cylinder"]
    FRICTIONS = [[0.4,0.0,0.0], [0.8,0.0,0.0], [1.2,0.0,0.0]]
    CONDIMS   = [1]  # add 3 if you want torsion+roll

    jobs = []

    gym.register_envs(gymnasium_robotics)

    for env_name in ENVS:
        # Probe geometry for THIS env (so sizes match that env’s object0)
        env = gym.make(env_name)
        env.reset()
        try:
            typ, sz = get_baseline_type_and_size(env, "object0")
        finally:
            env.close()

        baseline_mass = base.get("override_object", {}).get("mass", 2.0)
        base_run = base.get("run_name", "run")

        for agent in AGENTS:
            # ---------- 0) Baseline ----------
            cfg = deepcopy(base)
            cfg["env_name"] = env_name
            cfg["agent"]    = agent
            cfg["run_name"] = f"{base_run}__{env_name}__{agent}__baseline"
            cfg.pop("override_object", None)
            cfg.pop("override_friction", None)
            jobs.append(cfg)

            # ---------- 1) Shape sweep ----------
            if typ == "box":
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
                    jobs.append(cfg)
            else:
                # If baseline isn't a box, just echo that shape once
                cfg = deepcopy(base)
                cfg["env_name"] = env_name
                cfg["agent"]    = agent
                cfg["run_name"] = f"{base_run}__{env_name}__{agent}__shape_{typ}"
                cfg["override_object"] = {
                    "name": "object0",
                    "type": typ,
                    "size": [float(sz[0]), float(sz[1]), float(sz[2])],
                    "mass": float(baseline_mass),
                    "update_inertia": True
                }
                cfg.pop("override_friction", None)
                jobs.append(cfg)

            # ---------- 2) Friction sweep ----------
            for fv in FRICTIONS:
                for cd in CONDIMS:
                    cfg = deepcopy(base)
                    cfg["env_name"] = env_name
                    cfg["agent"]    = agent
                    tag = f"mu{fv[0]:.2f}_cd{cd}"
                    cfg["run_name"] = f"{base_run}__{env_name}__{agent}__fric_{tag}"
                    cfg["override_friction"] = {
                        "name":"object0",
                        "friction": fv,
                        "condim": cd
                    }
                    # Keep shape baseline
                    cfg.pop("override_object", None)
                    jobs.append(cfg)

    # Write configs
    for i, c in enumerate(jobs):
        with open(os.path.join(out_dir, f"cfg_{i:03d}.json"), "w") as f:
            json.dump(c, f, indent=2)

    # Make/update stable symlink sweeps/latest -> out_dir
    latest = os.path.join(BASE_DIR, "sweeps", "latest")
    try:
        if os.path.islink(latest) or os.path.exists(latest):
            os.remove(latest)
        os.symlink(out_dir, latest)
    except OSError:
        pass

    print(f"Wrote {len(jobs)} configs to: {out_dir}")
    print("Example:")
    print(f"  CFG_PATH={out_dir}/cfg_000.json python {BASE_DIR}/main_pars.py")
    print("Newest dir symlink:")
    print(f"  sweeps/latest -> {out_dir}")

if __name__ == "__main__":
    main()
