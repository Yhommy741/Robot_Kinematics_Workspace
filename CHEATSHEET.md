# FRA371 Kinematics Cheat Sheet (Workshop 0 + Labs 0–2 + DK / Statics)

> Units: revolute joints in **radians**, prismatic joints and lengths in **metres**.
> Short names: `s1 = sin(q1)`, `c12 = cos(q1 + q2)`.

### Which notebook has what

| Topic | Notebook | Section here |
|---|---|---|
| FK (symbolic MDH) | `fk.ipynb` | 3, 12 |
| IK by drawing | `ik_geometric.ipynb` | 5 |
| IK from the FK equations | `ik_algebraic.ipynb` | 5 |
| Jacobian (2 methods), manipulability | `fvk.ipynb` | 6, 7 |
| Joint speeds from tip speed, singularity | `ivk.ipynb` | 6, 7 |
| Joint torque from tip force | — | 8 |

## 0. Imports

```python
import time
import numpy as np
import sympy as sp
from pathlib import Path
from roboticstoolbox import Robot, DHRobot, RevoluteMDH, PrismaticMDH, jtraj
from spatialmath import SE3, SO3
import spatialgeometry as sg
import swift
np.set_printoptions(suppress=True)   # no 1e-17 junk when printing
```

---

## 0b. Making matrices & vectors (np / sp / SE3)

| Use | Library | Holds |
|---|---|---|
| Numbers, fast checks, RTB input/output | `np` (numpy) | floats |
| Formulas with `q1`, `l1`, … (FK, Jacobian by hand) | `sp` (sympy) | symbols |
| Poses / transforms (rotation + position) | `SE3`, `SO3` (spatialmath) | a valid 4×4 / 3×3 only |

### Create

| What | numpy | sympy | spatialmath |
|---|---|---|---|
| Matrix | `np.array([[1, 2], [3, 4]])` | `sp.Matrix([[1, 2], [3, 4]])` | — |
| Column vector | `np.array([x, y, z])` (1-D) | `sp.Matrix([x, y, z])` (3×1) | — |
| Identity | `np.eye(4)` | `sp.eye(4)` | `SE3()` |
| Zeros / ones | `np.zeros((3, 1))`, `np.ones(3)` | `sp.zeros(3, 1)`, `sp.ones(3, 1)` | — |
| Diagonal | `np.diag([1, 2, 3])` | `sp.diag(1, 2, 3)` | — |
| Rotation about z | build by hand | build by hand (`Rz` in the notebooks) | `SE3.Rz(q)`, `SE3.Rz(30, 'deg')`, `SO3.Rz(q)` |
| Translation | build by hand | build by hand | `SE3(x, y, z)`, `SE3.Trans([x, y, z])`, `SE3.Tx(d)` |
| From R and p | `T = np.eye(4); T[:3, :3] = R; T[:3, 3] = p` | `R.row_join(p).col_join(sp.Matrix([[0, 0, 0, 1]]))` | `SE3.Rt(R, p)` |
| Roll-pitch-yaw | — | — | `SE3.RPY([r, p, y], unit='deg')` |
| Symbols | — | `q1, q2 = sp.symbols('q1 q2', real=True)` | — |

### Use

| What | numpy | sympy | spatialmath |
|---|---|---|---|
| Multiply | `A @ B` (**not** `*`, that's element-wise) | `A * B` | `T1 * T2` |
| Transpose | `A.T` | `A.T` | — |
| Inverse | `np.linalg.inv(A)` | `A.inv()` | `T.inv()` |
| Determinant | `np.linalg.det(A)` | `A.det()` | — |
| Solve `A x = b` | `np.linalg.solve(A, b)` | `sp.solve(...)` / `A.LUsolve(b)` | — |
| Cross / dot | `np.cross(a, b)`, `a @ b` | `a.cross(b)`, `a.dot(b)` | — |
| Length | `np.linalg.norm(a)` | `a.norm()` | — |
| Eigen | `np.linalg.eigh(A)` (symmetric) / `eig` | `A.eigenvals()`, `A.eigenvects()` | — |
| Element | `A[0, 3]` | `A[0, 3]` | `T.A[0, 3]` |
| Rotation / position | `T[:3, :3]`, `T[:3, 3]` | `T[0:3, 0:3]`, `T[0:3, 3]` | `T.R`, `T.t` |
| Column j | `A[:, j]` | `A[:, j]` | `T.R[:, j]` |
| Stack side by side | `np.hstack([A, B])`, `np.c_[A, b]` | `sp.Matrix.hstack(A, B)`, `A.row_join(B)` | — |
| Stack on top | `np.vstack([A, B])`, `np.r_[a, b]` | `A.col_join(B)` | — |
| Size | `A.shape` | `A.shape` | — |
| Round | `np.round(A, 4)` | `A.evalf(4)` | `np.round(T.A, 4)` |

### Convert

| From → to | Code |
|---|---|
| sympy → numpy | `np.array(M.subs({q1: 0.5, l1: 0.64}).evalf(), dtype=float)` |
| sympy → numpy function (fast, many times) | `f = sp.lambdify((q1, q2), M.subs(vals), 'numpy')`, then `f(0.5, 0.2)` |
| numpy → sympy | `sp.Matrix(A)` |
| numpy 4×4 → SE3 | `SE3(A)` (must be a valid transform; for rounded/noisy numbers use `SE3(A, check=False)`) |
| SE3 → numpy | `T.A` (4×4), `T.R` (3×3), `T.t` (3,) |
| sympy → SE3 | `SE3(np.array(M.subs(vals).evalf(), dtype=float))` |

### Operators (same symbol, different meaning!)

| Operator | numpy array | sympy Matrix | SE3 |
|---|---|---|---|
| `A + B`, `A - B` | element-wise | element-wise | — |
| `A * B` | **element-wise** ⚠️ | **matrix multiply** | compose transforms |
| `A @ B` | matrix multiply ✅ | matrix multiply | — |
| `A * 2`, `A / 2` | scale every element | scale every element | — |
| `A ** 2` | **element-wise square** ⚠️ | matrix power `A·A` | `T ** 2` = `T * T` |
| `A ** -1` | element-wise `1/a` ⚠️ | inverse | — |
| Matrix power | `np.linalg.matrix_power(A, 2)` | `A ** 2` | — |
| Element-wise multiply | `A * B` | `A.multiply_elementwise(B)` | — |
| Apply a function to each element | `np.sin(A)` | `A.applyfunc(sp.sin)` | — |
| `-A` | negate | negate | — |
| `A == B` | element-wise bools → use `np.allclose(A, B)` | `True`/`False` (exact) → use `sp.simplify(A - B) == sp.zeros(...)` | `True`/`False` |
| Transform a point | `(T @ np.r_[p, 1])[:3]` | `(T * sp.Matrix([*p, 1]))[:3, 0]` | `T * p` (returns 3×1) |

Plain Python numbers: `7 / 2 = 3.5`, `7 // 2 = 3` (floor), `7 % 3 = 1` (remainder), `2 ** 3 = 8` (power, **not** `^`, which is XOR).

### Math functions

| Function | numpy (numbers) | sympy (symbols) |
|---|---|---|
| sin, cos, tan | `np.sin(x)` | `sp.sin(q1)` |
| asin, acos, atan | `np.arcsin`, `np.arccos`, `np.arctan` | `sp.asin`, `sp.acos`, `sp.atan` |
| atan2(y, x) | `np.arctan2(y, x)` | `sp.atan2(y, x)` |
| √ | `np.sqrt(x)` | `sp.sqrt(x)` |
| \|x\| | `np.abs(x)` | `sp.Abs(x)` |
| π | `np.pi` | `sp.pi` (exact) |
| deg ↔ rad | `np.deg2rad(30)`, `np.rad2deg(q)` | `sp.rad(30)`, `sp.deg(q)` |
| Exact fraction | — | `sp.Rational(1, 2)` (`1/2` in Python is already `0.5`) |
| Derivative | — | `sp.diff(expr, q1)`, `M.jacobian([q1, q2])` |
| Substitute | — | `expr.subs({q1: sp.pi/6})` |
| Simplify | — | `sp.simplify`, `sp.trigsimp`, `sp.expand`, `sp.factor` |
| To a float | `float(x)` | `float(expr.evalf())`, `expr.evalf(4)` |

- **Don't mix** `np.sin` with symbols (error) or `sp.sin` with arrays (slow): sympy for symbols, numpy for numbers.
- numpy 1-D `np.array([1, 2, 3])` has shape `(3,)`, not `(3, 1)`. `@` handles both; use `.reshape(3, 1)` if you need a column.

---

## 1. Load a robot from URDF

| Code | What it does | Inputs |
|---|---|---|
| `Path.cwd().as_posix()` | Project folder as a string | — |
| `Robot.URDF_read(path, tld=HERE)` | Read URDF (**older** RTB only) | `path`: urdf file, `tld`: folder that holds the package |
| `URDFRobot.URDF_read(full_path)` | Read URDF (**RTB ≥ 1.5**, what this venv has) | absolute path to `.urdf` |
| `update_package_cache({'pkg': folder})` | Tell RTB where `package://pkg/...` lives | dict: package name → folder |
| `Robot(links, name=name)` | Build the robot object | output of `URDF_read` |

```python
HERE = Path.cwd().as_posix()
if hasattr(Robot, 'URDF_read'):                       # older roboticstoolbox
    links, name = Robot.URDF_read('my_robot/robot.urdf', tld=HERE)[:2]
else:                                                  # roboticstoolbox 1.5+
    from roboticstoolbox.models.URDF import URDFRobot
    from xacrodoc.packages import update_package_cache
    update_package_cache({'my_robot': HERE + '/my_robot'})
    links, name = URDFRobot.URDF_read(HERE + '/my_robot/robot.urdf')[:2]
robot = Robot(links, name=name)
```

URDF mesh paths must be: `<mesh filename="package://my_robot/meshes/FILE.STL"/>`

---

## 2. Build a robot with Modified DH (MDH)

| Code | What it does | Inputs |
|---|---|---|
| `RevoluteMDH(a, alpha, d, offset, qlim)` | Rotating joint row, variable = θ | `a`, `d` [m]; `alpha`, `offset` [rad]; `qlim=[lo, hi]` |
| `PrismaticMDH(a, alpha, theta, offset, qlim)` | Sliding joint row, variable = d | `theta` fixed [rad]; `offset` [m] added to d |
| `DHRobot([rows...], name='x')` | Robot from MDH rows | list of joints |
| `robot.tool = SE3.Tz(l)` / `SE3(x, y, z)` | End-effector offset after last joint | SE3 |
| `robot.base = SE3.Tz(h)` | Where the robot sits in the world | SE3 |
| `print(robot)` | Show the DH table | — |
| `robot.n` | Number of joints | — |
| `robot.qlim` | Limits: row 0 = lower, row 1 = upper | — |
| `robot.q = q` | Set current joint values | array of length `n` |

MDH row = `Rx(alpha) · Tx(a) · Rz(theta) · Tz(d)`
Standard DH = `Rz(theta) · Tz(d) · Tx(a) · Rx(alpha)`

```python
# Lab 1 example (R-R-P robot)
robot_mdh = DHRobot([
    RevoluteMDH(a=0.0, alpha=0.0,     d=0.64),
    RevoluteMDH(a=0.0, alpha=np.pi/2, d=-0.2, offset=np.pi/2),
    PrismaticMDH(a=0.0, alpha=np.pi/2, offset=0.22),
], name='my_mdh')
robot_mdh.tool = SE3.Tz(0.095)
```

- Length of link *i* goes in `a` of row *i+1*; the last link goes in `robot.tool`.
- `offset` = value added to the joint variable so that q = 0 matches your drawing.

---

## 3. Forward kinematics (FK)

| Code | Returns | Inputs |
|---|---|---|
| `robot.fkine(q)` | End-effector pose `SE3` | `q`: joint values (len `n`) |
| `robot.fkine_all(q)` | Pose of every link frame | `q` |
| `np.deg2rad(x)` / `np.rad2deg(x)` | Degree ↔ radian | number or array |

### Reading an `SE3` result `T`

| Code | Gives |
|---|---|
| `T.t` | Position `[x, y, z]` |
| `T.R` | 3×3 rotation matrix |
| `T.A` | Full 4×4 numpy array |
| `T.R[:, 0]` / `[:, 1]` / `[:, 2]` | Tool x / y / z axis direction |
| `T.rpy(unit='deg')` | Roll, pitch, yaw |
| `T.eul(unit='deg')` | ZYZ Euler angles |
| `T.inv()` | Inverse transform |

```python
q = np.array([np.deg2rad(30), np.deg2rad(30), 0.6 * 0.3])   # R, R, P (60% of 0.3 m)
T = robot_mdh.fkine(q)
print(np.round(T.t, 4)); print(np.round(T.R, 4))
```

---

## 4. Making / combining transforms (`spatialmath.SE3`)

| Code | What it does | Inputs |
|---|---|---|
| `SE3(x, y, z)` | Pure translation | metres |
| `SE3(*P)` | Translation from a list/array `P=[x,y,z]` | array |
| `SE3(M)` | Wrap a 4×4 numpy matrix as SE3 | 4×4 array (must be a valid transform) |
| `SE3.Tx(d)` / `Ty` / `Tz` | Translate along one axis | metres |
| `SE3.Rx(θ)` / `Ry` / `Rz` | Rotate about one axis | radians, or `SE3.Rz(90, 'deg')` |
| `SE3.RPY([r, p, y], unit='deg')` | Rotation from roll-pitch-yaw | 3 angles |
| `A * B` | Chain transforms (B expressed in A's frame) | SE3 × SE3 |
| `T * SE3.Tx(1)` vs `SE3.Tx(1) * T` | Right = move in **local** frame, left = move in **world** frame | |

### Frame change with plain numpy (Lab 2 camera → base)

| Code | What it does |
|---|---|
| `np.linalg.inv(T)` | Inverse of a 4×4 transform |
| `np.append(v, 1)` | `[x, y, z]` → homogeneous `[x, y, z, 1]` |
| `v[:3]` | Back to `[x, y, z]` |
| `A @ B` | Matrix multiply (numpy arrays) |

```python
# T = base frame seen from camera, M_cam = object in camera frame
Pobj = (np.linalg.inv(T) @ np.append(M_cam, 1))[:3]
# same thing with SE3:
Pobj = (SE3(np.linalg.inv(T)) * SE3(M)).t
```

---

## 5. Inverse kinematics (IK)

| Code | What it does | Inputs |
|---|---|---|
| `robot.ikine_LM(Tep, q0=..., mask=...)` | Numerical IK (Levenberg–Marquardt) | see below |
| `sol.q` | Joint solution | — |
| `sol.success` | `True` if it converged | — |

`ikine_LM` main parameters:

| Param | Meaning | Default |
|---|---|---|
| `Tep` | Target pose, e.g. `SE3(*P)` | required |
| `q0` | Start guess — **changes which branch you get** (elbow up/down) | random |
| `mask` | Which of `[x, y, z, rx, ry, rz]` to match (1 = match). Planar xy only: `[1,1,0,0,0,0]` | all 6 |
| `ilimit` / `slimit` | Iterations per search / number of restarts | 30 / 100 |
| `tol` | Error tolerance | 1e-6 |
| `joint_limits` | Reject solutions outside `qlim` | `True` |

```python
sol = robot.ikine_LM(SE3(*Pobj), q0=[0, 0], mask=[1, 1, 0, 0, 0, 0])
print(np.round(sol.q, 4), sol.success)
robot.q = sol.q; env.step()
```

### 2-link planar arm by hand (L1, L2)

```
x = L1·cos(q1) + L2·cos(q1+q2)
y = L1·sin(q1) + L2·sin(q1+q2)
cos(q2) = (x² + y² − L1² − L2²) / (2·L1·L2)       → q2 = ±acos(...)  (2 branches)
q1 = atan2(y, x) − atan2(L2·sin(q2), L1 + L2·cos(q2))
```

Use `np.arctan2(y, x)`, `np.arccos(c)`.

### Two ways to solve IK by hand

| | Geometric | Algebraic |
|---|---|---|
| Start from | A drawing (top view + side view) | The FK equations `x = …, y = …, z = …` |
| Tools | Right triangles, angle of a vector | Square-and-add, `sin² + cos² = 1`, solve linear in `c, s` |
| Typical move | "the tip angle = q1 + angle of (ρ, l2)" | "x and y are linear in c1, s1 → 2×2 system" |

Algebraic recipe:
1. **Square and add** equations so angles cancel → the length / prismatic joint.
2. Find an equation with **one** angle → get `s` (or `c`), then `c = ±√(1 − s²)` → **2 branches**.
3. Remaining equations are **linear in `c_i, s_i`** → solve, then `q_i = atan2(s_i, c_i)`.

Form `a·cosθ + b·sinθ = d` → `θ = atan2(b, a) ± atan2(√(a² + b² − d²), d)`

### Lab R-R-P robot IK (l1 = 0.64, l2 = 0.2, l3 = 0.22, l4 = 0.095)

FK: `x = L·c1c2 − l2·s1`, `y = L·s1c2 + l2·c1`, `z = l1 + L·s2`, with `L = l3 + q3 + l4`

```
ρ  = ±√(x² + y² − l2²)                 (= L·c2, sign = branch; reachable only if x² + y² ≥ l2²)
q1 = atan2(ρ·y − l2·x, ρ·x + l2·y)     (= atan2(y, x) − atan2(l2, ρ))
q2 = atan2(z − l1, ρ)
q3 = √(x² + y² + (z − l1)² − l2²) − l3 − l4
```

- Use **`atan2`**, never `asin` / `acos` alone: they can't tell the two branches apart.
- Check: FK of your answer should give back `(x, y, z)`.

---

## 6. Differential (velocity) kinematics

Twist (lecture order, **angular on top**):

```
ξ = [ω; v] = J(q) · q̇        J = [Jω; Jv]       (6 × n)
```

### Two ways to get the Jacobian

**Method 1 – differentiate FK** (gives only `Jv`):

```
Jv = ∂p_e/∂q        column j = ∂[x, y, z]/∂q_j
```

2-link planar: `Jv = [[−l1s1 − l2s12, −l2s12], [l1c1 + l2c12, l2c12]]`

**Method 2 – geometric, column by column** (gives `Jω` and `Jv`):

| Joint j | `Jω,j` | `Jv,j` |
|---|---|---|
| Revolute (ρ = 1) | `ẑ_j` | `ẑ_j × (p_e − p_j)` |
| Prismatic (ρ = 0) | `0` | `ẑ_j` |

- `ẑ_j` = joint axis in base frame = column 3 of `⁰R_j` (`T0j[0:3, 2]`)
- `p_j` = joint position = column 4 of `⁰T_j` (`T0j[0:3, 3]`)
- With MDH, joint j sits on frame j. With standard DH, joint j sits on frame j−1.

**`ω ≠ ṙ`**: differentiating roll-pitch-yaw gives `ṙ`, not `ω`. Convert with `ω = J_rpy(r)·ṙ`, ZYX: `J_rpy = [[c1c2, −s1, 0], [s1c2, c1, 0], [−s2, 0, 1]]`.

### Forward / inverse velocity

| | Equation | Needs |
|---|---|---|
| FVK | `ξ = J·q̇` | nothing |
| IVK | `q̇ = J*⁻¹·ξ*` | square **reduced** `J*` (keep only the rows you control) and `det J* ≠ 0` |

- Reduced Jacobian: 2-link planar → rows `vx, vy` (2×2); Lab R-R-P → `Jv` (3×3).
- `A⁻¹ = adj(A) / det(A)`, `adj = (cofactor matrix)ᵀ`, cofactor `C_ij = (−1)^(i+j)·minor_ij`.

### Code

| Code | What it does |
|---|---|
| `robot.jacob0(q)` | 6×n Jacobian in world frame, **rows `[v; ω]`** (opposite of the lecture) |
| `robot.jacobe(q)` | Same, in the end-effector frame |
| `np.vstack([J[3:], J[:3]])` | RTB `[v; ω]` → lecture `[ω; v]` |
| `p.jacobian(q)` (SymPy) | Method 1 symbolically |
| `z.cross(p_e - p_j)` (SymPy) / `np.cross(a, b)` | Method 2 cross product |
| `np.linalg.solve(Jr, v)` | IVK, better than `inv(Jr) @ v` |
| `np.linalg.pinv(J)` | Pseudo-inverse when `J` is not square |

```python
J = robot.jacob0(q)
v, w = (J @ qd)[:3], (J @ qd)[3:]
qd = np.linalg.solve(J[:3], v_want)                  # IVK for a 3-joint arm (position only)
```

---

## 7. Singularity & manipulability

**Singularity**: `det J* = 0` → the arm loses a direction of motion, and IVK speeds go to ∞.

| Robot | `det J*` | Singular when |
|---|---|---|
| 2-link planar | `l1·l2·s2` | `q2 = 0, π` (arm straight or folded) |
| Lab R-R-P | `c2·L²` | `q2 = ±90°` (arm straight up / down) |

**Manipulability**: `M = √det(J Jᵀ)` (for a square `J`: `M = |det J|`). Bigger = moves more easily, 0 = singular.

- Use the **reduced** J (rows you care about). A 6×n `J` with n < 6 always gives `M = 0`.

**Manipulability ellipsoid**:
1. Find `J`
2. `A = J Jᵀ`
3. Eigen of `A`: `det(A − λI) = 0` → `λ`, then `(A − λI)x = 0` → `x`
4. Ellipsoid axis i points along `x_i` with half-length `√λ_i` (long = easy direction, 0 = stuck direction)

| Code | What it does |
|---|---|
| `robot.manipulability(q, axes='trans')` | `M` from the position rows only (use this for a 2–3 joint arm) |
| `np.sqrt(np.linalg.det(J @ J.T))` | `M` by hand |
| `lam, vec = np.linalg.eigh(A)` | Eigenvalues (ascending) + eigenvectors as **columns** of `vec` |
| `np.linalg.det(J)` | Determinant |

---

## 8. Statics (force → joint torque)

Wrench at the tip: `w = [f; n]` (force, moment). Joint effort `τ` (torque for R, force for P).

**Virtual work (fast way)**:

```
τ = Jᵀ · ⁰w            and if the force is given in the tip frame:   ⁰w = ⁰R_e · ᵉw
```

**Link by link (FBD way)**, from the tip back to the base:

```
ⁱf = ⁱR_{i+1} · ^{i+1}f
ⁱn = ⁱR_{i+1} · ^{i+1}n + ⁱP_{i+1} × ⁱf
τ_i = ⁱn · ẑ  (revolute)        τ_i = ⁱf · ẑ  (prismatic)
```

Worked example (slide 62): 2-link, `l1 = 1, l2 = 2, θ1 = θ2 = 45°`, `ᵉf = [3, 2]` N
→ `⁰f = R(90°)·[3, 2] = [−2, 3]` → `τ = Jᵀ⁰f = [[−2.707, 0.707], [−2, 0]]·[−2, 3] = [7.535, 4]` N·m

```python
J = robot.jacob0(q)                       # rows [v; ω] → wrench must be [f; n] in the same order
w = np.r_[T.R @ f_tip, T.R @ n_tip]       # tip frame → base frame
tau = J.T @ w
```

- Same `J` as velocity: fast directions for speed are weak directions for force.
- At a singularity the arm can hold some forces with **zero** torque.

---

## 9. Trajectory

| Code | What it does | Inputs |
|---|---|---|
| `jtraj(q0, q1, n)` | Smooth joint path; `.q` = n×joints array | start, end, number of steps |

```python
for qk in jtraj(q0, q1, 50).q:
    robot.q = qk; env.step(0.05)
```

---

## 10. Swift 3D viewer + shapes

| Code | What it does | Inputs |
|---|---|---|
| `env = swift.Swift()` | Create a viewer | — |
| `env.launch(realtime=True)` | Open in a browser tab | `realtime`: run at real speed |
| `env.launch(realtime=True, browser="notebook")` | Show inside the notebook cell (needs WebGL) | |
| `env.add(obj)` | Add a robot or shape | robot / `sg` shape |
| `env.step(dt)` | Redraw (call after every change!) | `dt` seconds, optional |
| `env.hold()` | Keep window open at end of a script | — |
| `env.close()` | Close viewer | — |

Shapes (`spatialgeometry as sg`) — **an `SE3` alone is not a shape**:

| Code | Inputs |
|---|---|
| `sg.Sphere(radius=0.03, pose=SE3(*P), color=(1, 0.1, 0.1, 0.6))` | radius [m], pose, color RGBA 0–1 |
| `sg.Cuboid(scale=[lx, ly, lz], pose=..., color=...)` | box side lengths [m] |
| `sg.Cylinder(radius=r, length=l, pose=..., color=...)` | radius, length [m] |
| `sg.Axes(length=0.1, pose=T)` | draws a coordinate frame at `T` |
| `shape.T = SE3(*P)` | move a shape after creating it |

```python
ball = sg.Sphere(radius=0.03, color=(1.0, 0.1, 0.1, 0.6))
ball.T = SE3(*Pobj)
env.add(ball); env.step()
```

---

## 11. Checking answers (numpy)

| Code | What it does |
|---|---|
| `np.round(x, 4)` | Round for printing |
| `np.linalg.norm(a - b)` | Distance / error between two vectors or matrices |
| `np.allclose(a, b, atol=1e-4)` | `True` if nearly equal |
| `np.zeros(n)` | Home config `q = [0, …, 0]` |
| `time.sleep(2)` | Pause so you can watch Swift |

```python
err = np.linalg.norm(robot_mdh.fkine(q).t - urdf_robot.fkine(q).t)   # MDH vs URDF (Lab 1)
```

---

## 12. SymPy (symbolic FK, Workshop 0)

| Code | What it does |
|---|---|
| `q1, q2 = sp.symbols('q1 q2', real=True)` | Symbolic variables |
| `sp.Matrix([...])`, `sp.eye(4)` | Matrix / identity |
| `sp.sin`, `sp.cos`, `sp.pi` | Use these on symbols, **not** `np.` |
| `A * B` | Matrix multiply |
| `sp.simplify(T)`, `T.applyfunc(sp.trigsimp)` | Clean up |
| `T[0:3, 3]`, `T[0:3, 0:3]` | Position / rotation |
| `T.subs({q1: sp.pi/2, ...}).evalf()` | Plug in numbers |
| `T[0:3, 3].jacobian([q1, q2])` | Symbolic Jacobian |
| `a.cross(b)`, `J.T`, `J.det()`, `J.inv()` | Cross product, transpose, determinant, inverse |
| `sp.factor(sp.simplify(J.det()))` | Clean determinant (shows singular factors like `sin(q2)`) |
| `sp.solve(eq, x)`, `sp.solve([e1, e2], [x, y])` | Solve one / several equations |
| `sp.Eq(lhs, rhs)` | Equation object for `solve` / display |
| `sp.lambdify((q1, q2), expr.subs(vals), 'numpy')` | Turn a formula into a fast numpy function |
| `np.array(T.subs(vals).evalf(), dtype=float)` | To numpy (compare with `robot.fkine(q).A`) |

---

## 13. Gotchas

- **Angles in radians.** `np.deg2rad(30)`, not `30`.
- **Always `env.step()`** after `robot.q = ...` or `env.add(...)`, otherwise Swift doesn't update. `env.step` without `()` does nothing.
- **Use the right robot variable** — `robot` vs `my_robot` vs `robot_mdh` are different robots.
- **`env.add(SE3(...))` shows nothing** — wrap it in `sg.Sphere(..., pose=SE3(...))`.
- **IK has several answers**; `q0` picks which one. Check `sol.success`.
- **By-hand IK: use `atan2`**, not `asin` / `acos` alone, or you lose a branch.
- **`jacob0` rows are `[v; ω]`**, the lecture writes `[ω; v]`. Swap before comparing, and give `τ = Jᵀw` the wrench as `[f; n]`.
- **`ω ≠ ṙ`**: differentiating orientation angles is not angular velocity.
- **Manipulability of a 6×n Jacobian with n < 6 is always 0** — use the reduced J or `axes='trans'`.
- **Force given in the tip frame** → rotate it to the base first: `⁰f = ⁰R_e · ᵉf`.
- **Prismatic limits**: "60% of joint limit" = `0.6 * qlim_upper` (in metres).
- **`Robot.URDF_read` is gone in RTB 1.5** — use the `URDFRobot` branch in section 1.
- **`ModuleNotFoundError: xacro`** → `pip install xacro`, then restart the kernel.
- **Swift in VS Code notebook times out** → VS Code has no WebGL; use `env.launch(realtime=True)` (browser tab) or start VS Code with `--enable-unsafe-swiftshader`.
