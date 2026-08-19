"""
4x ADXL345 disk -> angular acceleration, using ONLY the accelerometers.

Reads all four sensors over SPI, rotates each into the shared disk frame,
and solves the rigid-body equation

    a_i = a_c + (alpha x r_i) - wz^2 * r_i

for a_c (3), alpha (3) and wz^2 (1): 7 unknowns from 12 equations.
No gyro required for spin about the disk axis.

Run with --check first to verify the axis convention before trusting alpha.
"""

import argparse, csv, json, os, time
import numpy as np

# ------------------------- configuration -------------------------
CS_PINS      = [8, 7, 5, 6]              # BCM: CS1..CS4 -> header 24, 26, 29, 31
SENSOR_PHI   = {1: 90, 2: 0, 3: 270, 4: 180}   # mounting angle of each sensor
DISK_RADIUS  = 0.128                     # metres (0.256 m diameter)
G_TO_MS2     = 9.80665
LOG_FILE     = "disk_alpha_log.csv"
CALIB_FILE   = "disk_calib.json"

# The ADXL345 axes are right-handed, so with local +y radial-outward and
# local +z up, local +x must run tangentially CLOCKWISE. If your boards are
# actually mounted the other way round, flip this to -1.
# Use --check to determine the correct value experimentally.
TANGENTIAL_SIGN = +1

DEVID, POWER_CTL, DATA_FORMAT, DATAX0 = 0x00, 0x2D, 0x31, 0x32
READ_BIT, MULTI_BIT = 0x80, 0x40
MAX_PLAUSIBLE_G = 6.0


# ------------------------- per-sensor calibration -------------------------
# Each ADXL345 has its own zero-g offset and gain error (datasheet allows
# +/-150 mg). On a 0.128 m disk, 0.1 g of bias fakes ~8 rad/s^2 of alpha,
# so calibration is not optional. Run:  python3 disk_alpha.py --calibrate

def fit_ellipsoid(samples):
    """Fit per-axis offset and scale so |a| = 1 g at every orientation.

    Model   A(x-ox)^2 + B(y-oy)^2 + C(z-oz)^2 = 1,  linearised as
            A x^2 + B y^2 + C z^2 + D x + E y + F z = 1  (D = -2A*ox, ...)
    """
    s = np.asarray(samples, dtype=float)
    x, y, z = s[:, 0], s[:, 1], s[:, 2]
    M = np.column_stack([x*x, y*y, z*z, x, y, z])
    p, *_ = np.linalg.lstsq(M, np.ones(len(s)), rcond=None)
    A, B, C, D, E, F = p
    if min(A, B, C) <= 0:
        raise ValueError("fit failed - tumble through more orientations")
    off = np.array([-D/(2*A), -E/(2*B), -F/(2*C)])
    S = 1 + A*off[0]**2 + B*off[1]**2 + C*off[2]**2
    return off, np.sqrt(S/np.array([A, B, C]))


def load_calib():
    if not os.path.exists(CALIB_FILE):
        return None
    with open(CALIB_FILE) as f:
        d = json.load(f)
    return {int(k): (np.array(v["offset"]), np.array(v["scale"])) for k, v in d.items()}


def load_alignment():
    """Measured local->disk rotations, if calibration recorded them."""
    if not os.path.exists(CALIB_FILE):
        return None
    with open(CALIB_FILE) as f:
        d = json.load(f)
    if not all("rotation" in v for v in d.values()):
        return None
    return {int(k): np.array(v["rotation"]) for k, v in d.items()}


def apply_calib(raw, calib, sid):
    if calib is None or sid not in calib:
        return raw
    off, scale = calib[sid]
    return (raw - off) / scale


def _angle_between(u, v):
    u = u / np.linalg.norm(u); v = v / np.linalg.norm(v)
    return float(np.degrees(np.arccos(np.clip(u @ v, -1.0, 1.0))))


def _validate_fit(off, scale):
    """The ellipsoid fit has 6 free parameters. Given too few distinct
    directions it will match those points perfectly with absurd parameters,
    so sanity-check the result rather than trusting a low residual."""
    if np.any(scale < 0.8) or np.any(scale > 1.2):
        return f"scale {np.round(scale,3)} outside 0.8-1.2"
    if np.any(np.abs(off) > 0.35):
        return f"offset {np.round(off,3)} exceeds 0.35 g"
    return None


def _kabsch(A, B):
    """Rotation R minimising ||R @ a_i - b_i||, i.e. mapping frame A onto B."""
    H = A.T @ B
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def fit_alignment(kept):
    """Measure each board's ACTUAL orientation instead of assuming ideal.

    Boards are never mounted at perfect 90 deg steps in a perfect plane -- a
    few degrees of in-plane and out-of-plane error is normal, and out-of-plane
    error is invisible to any rotation about the disk axis, so it cannot be
    fixed by choosing a different SENSOR_PHI.

    Sensor 1's nominal frame defines the disk frame; sensors 2-4 are then
    aligned to it from the static poses via Kabsch. A residual error common
    to all four just re-defines the disk frame slightly, which is harmless --
    what matters is that the four agree with each other.
    """
    ref = np.array(kept[1])
    out = {1: sensor_rotation(SENSOR_PHI[1])}
    report = {}
    for s in sorted(kept):
        if s == 1:
            report[s] = (0.0, float(np.abs(np.array(kept[1]) - ref).mean()))
            continue
        loc = np.array(kept[s])
        R_rel = _kabsch(loc, ref)
        before = float(np.abs(loc - ref).mean())
        after = float(np.abs(np.array([R_rel @ v for v in loc]) - ref).mean())
        out[s] = out[1] @ R_rel
        nominal = sensor_rotation(SENSOR_PHI[1]).T @ sensor_rotation(SENSOR_PHI[s])
        dev = np.degrees(np.arccos(np.clip((np.trace(R_rel @ nominal.T) - 1) / 2, -1, 1)))
        report[s] = (float(dev), after)
    return out, report


def calibrate(spi, GPIO, n_poses=14, still_window=0.6, still_tol=0.010,
              hold=1.0, min_sep=25.0):
    """Static multi-pose calibration.

    Hand-tumbling does not work: the sensors sit at r = DISK_RADIUS, so any
    rotation adds w^2*r and alpha*r of IN-PLANE acceleration -- tens of milli-g
    even when moving gently. That breaks the "gravity is the only acceleration"
    assumption, and shows as x/y scales inflated above 1.0.

    So: place the disk in a pose, hold still, let it capture, then move on.
    Each new pose must point gravity at least min_sep degrees away from every
    pose already captured, otherwise the fit is underdetermined.
    """
    print(f"\n  STATIC POSE CALIBRATION - target {n_poses} poses")
    print(f"  stillness < {still_tol:.3f} g, poses must differ by > {min_sep:.0f} deg")
    print("  Place the disk, hold dead still, wait for capture, then move to a")
    print("  GENUINELY DIFFERENT orientation - flat, upside down, on each edge,")
    print("  propped at odd angles. Repeating a pose does not help the fit.")
    print("  Ctrl+C when done (10+ well-spread poses).\n")

    kept = {s: [] for s in SENSOR_PHI}
    ref = []          # sensor-1 gravity directions already captured
    try:
        while len(ref) < n_poses:
            recent = {s: [] for s in SENSOR_PHI}
            need = max(3, int(still_window / 0.02))
            while True:
                for i, pin in enumerate(CS_PINS):
                    g = read_accel_g(spi, GPIO, pin)
                    if g is not None:
                        recent[i + 1].append(g)
                        if len(recent[i + 1]) > need:
                            recent[i + 1].pop(0)
                time.sleep(0.02)
                full = [np.array(v) for v in recent.values() if len(v) >= need]
                if len(full) < len(CS_PINS):
                    continue
                motion = max(float(a.std(axis=0).max()) for a in full)
                if motion >= still_tol:
                    print(f"\r  pose {len(ref)+1}/{n_poses}: hold still "
                          f"(motion {motion:.4f} g, need < {still_tol:.3f})      ",
                          end="", flush=True)
                    continue
                cur = np.array(recent[1]).mean(axis=0)
                sep = min([_angle_between(cur, r) for r in ref], default=999.0)
                if sep < min_sep:
                    print(f"\r  pose {len(ref)+1}/{n_poses}: too close to a pose "
                          f"already captured ({sep:.0f} deg) - reorient      ",
                          end="", flush=True)
                    continue
                break

            grab = {s: [] for s in SENSOR_PHI}
            t0 = time.monotonic()
            while time.monotonic() - t0 < hold:
                for i, pin in enumerate(CS_PINS):
                    g = read_accel_g(spi, GPIO, pin)
                    if g is not None:
                        grab[i + 1].append(g)
                time.sleep(0.02)

            if any(len(v) < 10 or np.array(v).std(axis=0).max() > still_tol * 2
                   for v in grab.values()):
                print(f"\r  pose {len(ref)+1}/{n_poses}: moved during capture, retry   ",
                      end="", flush=True)
                continue

            for s, v in grab.items():
                kept[s].append(np.array(v).mean(axis=0))
            ref.append(kept[1][-1])
            g1 = ref[-1]
            print(f"\r  pose {len(ref)}/{n_poses} captured   sensor1 = "
                  f"[{g1[0]:+.3f} {g1[1]:+.3f} {g1[2]:+.3f}]                    ")
    except KeyboardInterrupt:
        print("\n  stopped early")

    n = len(ref)
    print()
    if n < 8:
        print(f"  Only {n} distinct poses - the fit needs at least 8 (ideally 12+).")
        print("  Nothing saved; the previous calibration is untouched.")
        return

    out, failed = {}, []
    for s in sorted(kept):
        arr = np.array(kept[s])
        before = np.linalg.norm(arr, axis=1)
        try:
            off, scale = fit_ellipsoid(arr)
        except ValueError as e:
            print(f"  sensor {s}: {e}")
            failed.append(s); continue
        bad = _validate_fit(off, scale)
        if bad:
            print(f"  sensor {s}: REJECTED - {bad}")
            print(f"            (offset={np.round(off,4)} scale={np.round(scale,4)})")
            failed.append(s); continue
        after = np.linalg.norm((arr - off) / scale, axis=1)
        print(f"  sensor {s}: offset={np.round(off,4)} scale={np.round(scale,4)}")
        print(f"            |a| error  before {np.abs(before-1).mean():.4f} g"
              f"  ->  after {np.abs(after-1).mean():.4f} g")
        out[s] = {"offset": off.tolist(), "scale": scale.tolist()}

    if failed:
        print(f"\n  Sensors {failed} failed validation, so NOTHING was saved.")
        print("  Almost always this means too few genuinely different orientations.")
        print("  Aim for 12+ poses spread over the whole sphere: each face down,")
        print("  each edge down, and several tilted in between.")
        return

    # Apply the offsets we just fitted, then measure real board orientations
    cal = {s: (np.array(v["offset"]), np.array(v["scale"])) for s, v in out.items()}
    corrected = {s: [apply_calib(v, cal, s) for v in kept[s]] for s in kept}
    align, report = fit_alignment(corrected)
    print("\n  Measured board orientations (vs the ideal 90 deg spacing):")
    for s in sorted(report):
        dev, res = report[s]
        note = "  <-- large, check the mounting" if dev > 10 else ""
        print(f"    sensor {s}: off nominal by {dev:5.2f} deg, "
              f"residual {res:.4f} g{note}")
    for s in out:
        out[s]["rotation"] = align[s].tolist()

    with open(CALIB_FILE, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved to {CALIB_FILE} from {n} distinct static poses.")


# ------------------------- geometry -------------------------
def sensor_rotation(phi_deg, tan_sign=TANGENTIAL_SIGN):
    """local (x=tangential, y=radial out, z=up) -> disk frame. det must be +1."""
    p = np.radians(phi_deg)
    x_axis = tan_sign * np.array([ np.sin(p), -np.cos(p), 0.0])   # tangential
    y_axis =            np.array([ np.cos(p),  np.sin(p), 0.0])   # radial out
    z_axis =            np.array([ 0.0,        0.0,       1.0])
    return np.column_stack([x_axis, y_axis, z_axis])

def sensor_positions(radius, phi_degs):
    return {s: radius * np.array([np.cos(np.radians(p)), np.sin(np.radians(p)), 0.0])
            for s, p in phi_degs.items()}

def skew(v):
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])

R_MAT = {s: sensor_rotation(p) for s, p in SENSOR_PHI.items()}
POS   = sensor_positions(DISK_RADIUS, SENSOR_PHI)

def to_disk_frame(sid, a_local):
    return R_MAT[sid] @ np.asarray(a_local, dtype=float)

# ------------------------- solver -------------------------
def solve(accels_ms2):
    """accels_ms2: {sid: 3-vector in disk frame, m/s^2}
    returns a_c, alpha, wz2, residual_rms"""
    A_rows, b_rows = [], []
    for sid, a in accels_ms2.items():
        r = POS[sid]
        A_rows.append(np.hstack([np.eye(3), -skew(r), (-r).reshape(3, 1)]))
        b_rows.append(np.asarray(a, dtype=float))
    A, b = np.vstack(A_rows), np.concatenate(b_rows)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    return x[:3], x[3:6], float(x[6]), float(np.sqrt(np.mean((A @ x - b) ** 2)))

# ------------------------- SPI -------------------------
def setup_spi():
    import spidev, RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    for pin in CS_PINS:
        GPIO.setup(pin, GPIO.OUT, initial=GPIO.HIGH)
    spi = spidev.SpiDev()
    spi.open(0, 0)
    spi.no_cs = True
    spi.max_speed_hz = 1000000
    spi.mode = 0b11
    return spi, GPIO

def read_reg(spi, GPIO, pin, reg, length=1):
    GPIO.output(pin, GPIO.LOW)
    cmd = reg | READ_BIT | (MULTI_BIT if length > 1 else 0)
    out = spi.xfer2([cmd] + [0x00] * length)
    GPIO.output(pin, GPIO.HIGH)
    return out[1:]

def write_reg(spi, GPIO, pin, reg, value):
    GPIO.output(pin, GPIO.LOW)
    spi.xfer2([reg, value])
    GPIO.output(pin, GPIO.HIGH)

def read_accel_g(spi, GPIO, pin):
    raw = read_reg(spi, GPIO, pin, DATAX0, 6)
    def s16(lo, hi):
        v = (hi << 8) | lo
        return v - 65536 if v > 32767 else v
    v = np.array([s16(raw[0], raw[1]), s16(raw[2], raw[3]), s16(raw[4], raw[5])]) * 0.0039
    return None if np.max(np.abs(v)) > MAX_PLAUSIBLE_G else v

def init_sensors(spi, GPIO):
    ok = True
    for i, pin in enumerate(CS_PINS):
        d = read_reg(spi, GPIO, pin, DEVID)[0]
        good = (d == 0xE5)
        ok &= good
        print(f"  Sensor{i+1} (CS {pin}): DEVID=0x{d:02X} {'OK' if good else 'MISMATCH'}")
    for pin in CS_PINS:
        write_reg(spi, GPIO, pin, DATA_FORMAT, 0x08)
        write_reg(spi, GPIO, pin, POWER_CTL, 0x08)
    return ok

def sample(spi, GPIO, calib=None, align=None, want_raw=False):
    """returns {sid: disk-frame vector in m/s^2} or None if any sensor messed up

    With want_raw=True also returns the per-sensor readings in g, so the raw
    x/y/z behind every solve can be logged and re-analysed offline:
        raw[sid]  -- straight off the chip, no calibration
        cal[sid]  -- after offset/scale correction, still the board's own axes
    """
    out, raw, cal = {}, {}, {}
    for i, pin in enumerate(CS_PINS):
        g = read_accel_g(spi, GPIO, pin)
        if g is None:
            return (None, None, None) if want_raw else None
        sid = i + 1
        v = apply_calib(g, calib, sid)
        R = align[sid] if align and sid in align else R_MAT[sid]
        out[sid] = (R @ v) * G_TO_MS2
        raw[sid], cal[sid] = g, v
    return (out, raw, cal) if want_raw else out

# ------------------------- axis-convention check -------------------------
def _rot_z(theta_deg, tan_sign):
    """Candidate local->disk rotation: in-plane angle + handedness."""
    p = np.radians(theta_deg)
    x_axis = tan_sign * np.array([np.sin(p), -np.cos(p), 0.0])
    y_axis =            np.array([np.cos(p),  np.sin(p), 0.0])
    z_axis =            np.array([0.0, 0.0, 1.0])
    return np.column_stack([x_axis, y_axis, z_axis])


def axis_check(spi, GPIO, seconds=8.0):
    """Hold the disk STILL and TILTED (30-45 deg).

    With the disk stationary, gravity is the ONLY acceleration, and it is
    identical everywhere on a rigid disk. So after rotating each sensor's
    reading into the disk frame, all four must report the SAME vector.

    This searches every plausible mounting for each board independently
    (4 inplane orientations x 2 handedness) and reports the combination
    that makes the four sensors agree.
    """
    print("\n  Tilt the disk 30-45 deg and hold it STILL.")
    print(f"  Sampling for {seconds:.0f}s...\n")

    calib = load_calib()
    if calib:
        print("  (using saved calibration)")
    acc = {s: [] for s in SENSOR_PHI}
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        for i, pin in enumerate(CS_PINS):
            g = read_accel_g(spi, GPIO, pin)
            if g is not None:
                acc[i + 1].append(apply_calib(g, calib, i + 1))
        time.sleep(0.02)

    means, noise = {}, {}
    for s, v in acc.items():
        if v:
            arr = np.array(v)
            means[s] = arr.mean(axis=0)
            noise[s] = float(arr.std(axis=0).mean())

    print("  Mean raw reading per sensor (g), and sample noise:")
    for s in sorted(means):
        mag = np.linalg.norm(means[s])
        flag = "" if abs(mag - 1.0) < 0.12 else "   <-- |a| far from 1g, check this board"
        print(f"    sensor {s}: {np.round(means[s], 4)}   |a|={mag:.3f}g  noise={noise[s]:.4f}g{flag}")

    if len(means) < 4:
        print("\n  Fewer than 4 sensors responded check and fix wiring first.")
        return

    tilts = [np.degrees(np.arctan2(np.hypot(v[0], v[1]), v[2])) for v in means.values()]
    print(f"\n  Tilt during sampling: {np.mean(tilts):.1f} deg  ("
          + ", ".join(f"{t:.1f}" for t in tilts) + ")")
    if np.mean(tilts) < 15.0:
        print("\n  ABORT: the disk was nearly FLAT. Gravity then lies entirely")
        print("  along z, which no in-plane rotation affects; so there is nothing")
        print("  here to identify the mounting, and any answer would be noise.")
        print("  Prop the disk at 30-45 deg and run/check again.")
        return

    align = load_alignment()
    if align:
        vecs = np.array([align[s] @ means[s] for s in sorted(means)])
        spread = float(np.mean(np.std(vecs, axis=0)))
        g = vecs.mean(axis=0)
        print(f"\n  Using MEASURED board orientations from {CALIB_FILE}:")
        print(f"    gravity in disk frame = {np.round(g,4)} g  |g|={np.linalg.norm(g):.4f}")
        print(f"    disagreement between sensors = {spread:.4f} g")
        if spread < 0.01:
            print("    ==> good. Sensors agree to about the noise floor.")
        else:
            print("    ==> still high. Check for a loose board or one out of plane.")
        return

    sids = sorted(means)
    options = [(th, sg) for th in (0, 90, 180, 270) for sg in (+1, -1)]

    # For each sensor, its reading rotated by each candidate mounting
    cand = {s: {o: _rot_z(o[0], o[1]) @ means[s] for o in options} for s in sids}

    # Sensor 1 defines the reference frame; search the rest against it.
    best = None
    for o1 in options:
        for o2 in options:
            for o3 in options:
                for o4 in options:
                    combo = (o1, o2, o3, o4)
                    vecs = np.array([cand[s][o] for s, o in zip(sids, combo)])
                    spread = float(np.mean(np.std(vecs, axis=0)))
                    if best is None or spread < best[0]:
                        best = (spread, combo, vecs.mean(axis=0))

    spread, combo, gvec = best
    print(f"\n  Best-fit mounting (disagreement {spread:.4f} g):")
    for s, (th, sg) in zip(sids, combo):
        print(f"    sensor {s}: in-plane angle {th:3d} deg, TANGENTIAL_SIGN {sg:+d}"
              f"   (config says {SENSOR_PHI[s]:3d} deg, {TANGENTIAL_SIGN:+d})")
    print(f"\n  Recovered gravity in disk frame: {np.round(gvec,4)} g  |g|={np.linalg.norm(gvec):.4f}")

    cfg = tuple((SENSOR_PHI[s], TANGENTIAL_SIGN) for s in sids)
    vecs_cfg = np.array([cand[s][o] for s, o in zip(sids, cfg)])
    spread_cfg = float(np.mean(np.std(vecs_cfg, axis=0)))
    print(f"  Your current config gives disagreement: {spread_cfg:.4f} g")

    if spread > 0.05:
        print("\n  WARNING: even the best fit disagrees badly. Likely causes:")
        print("    - the disk moved during sampling (must be perfectly still)")
        print("    - a board is tilted out of the disk plane, not just rotated")
        print("    - a sensor is faulty or miscalibrated")
    elif combo == cfg:
        print("\n  ==> Your current configuration is CORRECT.")
    else:
        signs = {sg for _, sg in combo}
        print("\n  ==> Update the config to the best-fit values above.")
        if len(signs) == 1:
            print(f"      Set TANGENTIAL_SIGN = {signs.pop():+d}")
            print(f"      Set SENSOR_PHI = {{" +
                  ", ".join(f"{s}: {th}" for s, (th, _) in zip(sids, combo)) + "}")
        else:
            print("      NOTE: boards have MIXED handedness, which means they are")
            print("      not all mounted the same way round. Fix that physically -")
            print("      per-sensor handedness is a mounting error, not a setting.")

# ------------------------- main -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="run the axis-convention check and exit")
    ap.add_argument("--calibrate", action="store_true", help="run per-sensor calibration and exit")
    ap.add_argument("--seconds", type=float, default=None, help="stop after N seconds")
    ap.add_argument("--still-tol", type=float, default=0.010,
                    help="stillness threshold in g for --calibrate (default 0.010)")
    ap.add_argument("--poses", type=int, default=14, help="target pose count for --calibrate")
    args = ap.parse_args()

    for s, p in SENSOR_PHI.items():
        d = np.linalg.det(sensor_rotation(p))
        assert abs(d - 1.0) < 1e-9, f"sensor {s} rotation det={d}, not a real rotation"
    print(f"Disk radius {DISK_RADIUS} m, TANGENTIAL_SIGN {TANGENTIAL_SIGN:+d}, rotations valid (det=+1)")

    spi, GPIO = setup_spi()
    try:
        if not init_sensors(spi, GPIO):
            print("\n  Not all sensors responded - fix wiring before trusting alpha.\n")
        if args.calibrate:
            calibrate(spi, GPIO, n_poses=args.poses, still_tol=args.still_tol)
            return
        if args.check:
            axis_check(spi, GPIO)
            return

        calib = load_calib()
        align = load_alignment()
        print("  calibration: " + ("loaded from " + CALIB_FILE if calib
                                   else "NONE - run --calibrate first, alpha will be biased"))
        print("  orientation: " + ("measured per board" if align
                                   else "ideal 90 deg assumed - re-run --calibrate to measure"))
        rows = []
        t0 = time.monotonic()
        print("\n  t(s)     alpha_x   alpha_y   alpha_z   |wz|    resid")
        while True:
            t = time.monotonic() - t0
            if args.seconds and t > args.seconds:
                break
            accels, raw, cal = sample(spi, GPIO, calib, align, want_raw=True)
            if accels is None:
                time.sleep(0.02); continue
            a_c, alpha, wz2, resid = solve(accels)
            wz = np.sqrt(max(wz2, 0.0))
            print(f"  {t:6.2f}  {alpha[0]:+8.3f}  {alpha[1]:+8.3f}  {alpha[2]:+8.3f}  {wz:6.3f}  {resid:6.3f}")
            row = [f"{t:.4f}", *np.round(alpha, 6), round(wz, 6), round(resid, 6),
                   *np.round(a_c, 6)]
            for s in sorted(raw):
                row += [*np.round(raw[s], 5), *np.round(cal[s], 5),
                        *np.round(accels[s], 5)]
            rows.append(row)
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        GPIO.cleanup(); spi.close()
        if not args.check and not args.calibrate and rows:
            with open(LOG_FILE, "w", newline="") as f:
                w = csv.writer(f)
                hdr = ["t_sec", "alpha_x", "alpha_y", "alpha_z", "omega_z_abs",
                       "residual_rms", "a_c_x", "a_c_y", "a_c_z"]
                for s in sorted(SENSOR_PHI):
                    hdr += [f"s{s}_raw_x", f"s{s}_raw_y", f"s{s}_raw_z",
                            f"s{s}_cal_x", f"s{s}_cal_y", f"s{s}_cal_z",
                            f"s{s}_disk_x", f"s{s}_disk_y", f"s{s}_disk_z"]
                w.writerow(hdr)
                w.writerows(rows)
            print(f"\nLogged {len(rows)} rows to {LOG_FILE}")

if __name__ == "__main__":
    main()
