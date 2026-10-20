"""食材の小片（凸多面体）を「切る」ことで作る形状生成器.

元の形（にんじん = 細い円柱、じゃがいも = 楕円体、大根 = 太い円柱）を半空間の集まりで
近似し、切断面（半空間）を足して交差を取ると、凸な小片が得られる。
交差の計算は scipy の HalfspaceIntersection で行う（元の形も切断面も凸なので結果も凸）。

切り方（食材ごとに ``FOOD_STYLES`` で決める）:

- ``random`` （乱切り。にんじん）: 小片の中心のまわりを 6〜12 枚のランダムな平面で切る
- ``dice``   （角切り。にんじん）: 直交する 3 組の平面で切る（向きはランダム）
- ``rounded``（丸みのある乱切り。じゃがいも）: 小さな楕円体（芋の塊）を 2〜3 枚の平面で切る。
  表面の多くが丸いまま残る
- ``ichou``  （いちょう切り。大根）: 太い円柱を輪切りにし、さらに中心を通る 2 枚の平面で
  4 等分した扇形の厚切り

出来た小片は次を満たすものだけを残す（満たさなければ作り直す）。

- 最も薄い方向の厚み（最小幅）が ``min_width`` 以上（薄切りは対象外）
- 最も長い寸法が ``max_extent`` 以下（グリッパに収まる）

最後に、机に置いたときに安定する向き（安定姿勢）に回し、外接箱の一番薄い軸が鉛直
になるようにそろえる（so101-nexus は外接箱の一番薄い軸を上にして置くため）。
安定姿勢でも、重心が接地面の端に近いとシミュレーションの中で転がり出すことがあるので、
重心が接地面の端から ``min_support_margin`` 以上内側にある向きだけを使う。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import trimesh
from scipy.spatial import ConvexHull, HalfspaceIntersection

FOOD_TYPES = ("carrot", "potato", "daikon")
CUT_STYLES = ("random", "dice", "rounded", "ichou")
# 食材ごとの切り方。セットの i 番目の片は FOOD_STYLES[食材][i % 個数] で切る
# （にんじんは乱切り:角切り = 2:1）
FOOD_STYLES = {
    "carrot": ("random", "random", "dice"),
    "potato": ("rounded",),
    "daikon": ("ichou",),
}
# 食材ごとの密度 [kg/m^3]（生の野菜の目安。じゃがいもは水に沈み、大根はほぼ水と同じ）
FOOD_DENSITY = {"carrot": 1040.0, "potato": 1090.0, "daikon": 970.0}


@dataclass(frozen=True)
class PieceSpec:
    min_width: float = 0.015  # 最小の厚み [m]
    max_extent: float = 0.035  # 最大の寸法 [m]
    cut_offset: tuple[float, float] = (0.008, 0.016)  # 中心から切断面までの距離 [m]
    # 重心（を真下に下ろした点）から接地面の端までの距離の下限 [m]。これより端に近いと、
    # 置いた直後に転がったり回ったりする（約 2mm 以下の片の 9 割で起きた）
    min_support_margin: float = 0.0035


@dataclass(frozen=True)
class PieceInfo:
    """書き出した小片のメタ情報."""

    name: str
    food: str
    style: str
    volume_cm3: float
    mass_kg: float
    extents_mm: tuple[float, float, float]  # 外接箱（x, y, z）。z が高さ
    min_width_mm: float
    n_vertices: int

    def to_dict(self) -> dict:
        return asdict(self)


def _base_halfspaces(food: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """乱切り・角切りの元の形を、半空間 ``A x + b <= 0`` の集まり（scipy の形式 [A | b]）と
    内部の点の範囲で返す（にんじん = 半径 11〜16mm の円柱）."""
    if food != "carrot":
        raise ValueError(f"random / dice cuts are defined for carrot only: {food}")
    r = rng.uniform(0.011, 0.016)
    half_len = 0.06
    ang = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    normals = np.column_stack([np.zeros_like(ang), np.cos(ang), np.sin(ang)])  # 軸は x
    offsets = np.full(len(ang), r)
    normals = np.vstack([normals, [[1, 0, 0], [-1, 0, 0]]])
    offsets = np.concatenate([offsets, [half_len, half_len]])
    inner = np.array([half_len - 0.02, r * 0.4, r * 0.4])
    return np.column_stack([normals, -offsets]), inner


def _polyhedron_normals() -> list[np.ndarray]:
    cube = np.vstack([np.eye(3), -np.eye(3)])
    signs = (1, -1)
    octa = np.array([[sx, sy, sz] for sx in signs for sy in signs for sz in signs]) / np.sqrt(3)
    ico = trimesh.creation.icosahedron().vertices  # 正十二面体の面の向き = 正二十面体の頂点
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    return [cube, octa, ico]


_POLY_NORMALS = _polyhedron_normals()


def _random_unit_vectors(rng: np.random.Generator, n: int) -> np.ndarray:
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def _ellipsoid_halfspaces(axes: np.ndarray, subdivisions: int) -> np.ndarray:
    """楕円体（半軸 ``axes``、中心が原点）を接平面の半空間の集まりで近似する."""
    sphere = trimesh.creation.icosphere(subdivisions=subdivisions).vertices
    points = sphere * axes  # 楕円体の表面の点
    normals = sphere / axes  # その点での外向き法線（未正規化）
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    offsets = np.einsum("ij,ij->i", normals, points)
    return np.column_stack([normals, -offsets])


def _rounded_halfspaces(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """丸みのある乱切り（じゃがいも）: 小さな楕円体を 2〜3 枚の平面で切る. 内部の点も返す."""
    axes = np.array([rng.uniform(0.0145, 0.0175), rng.uniform(0.0125, 0.0150), 0.0])
    axes[2] = rng.uniform(0.0105, min(0.0130, axes[1]))
    body = _ellipsoid_halfspaces(axes, subdivisions=2)
    # 机に置く面になるよう、一番薄い軸（z）の上か下を 1 枚平らに切る
    side = rng.choice([-1.0, 1.0])
    cuts = [[0.0, 0.0, side, -rng.uniform(0.45, 0.75) * axes[2]]]
    # 残りの 1〜2 枚は斜めの向きで、端を少しだけ切り落とす（切り口の平らな面）
    for n in _random_unit_vectors(rng, int(rng.integers(1, 3))):
        if abs(n[2]) > 0.8:  # 机に置く面と近い向きは避ける
            n = np.array([n[0], n[1], 0.0]) / max(np.linalg.norm(n[:2]), 1e-9)
        support = np.linalg.norm(axes * n)  # 楕円体のその向きへの張り出し
        cuts.append([*n, -rng.uniform(0.55, 0.85) * support])
    return np.vstack([body, np.array(cuts)]), np.zeros(3)


def _ichou_halfspaces(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """いちょう切り（大根）: 半径 r の円柱を厚さ t で輪切りにし、中心角 90° 前後の扇形にする."""
    r = rng.uniform(0.022, 0.028)
    t = rng.uniform(0.015, 0.020)
    span = np.radians(rng.uniform(85.0, 95.0))
    ang = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    normals = np.column_stack([np.cos(ang), np.sin(ang), np.zeros_like(ang)])  # 軸は z
    halfspaces = [np.column_stack([normals, np.full(len(ang), -r)])]
    halfspaces.append(np.array([[0, 0, 1, -t / 2], [0, 0, -1, -t / 2]]))  # 輪切りの 2 面
    # 中心を通る 2 枚の平面で扇形に切る（角度 0 と span の間を残す）
    halfspaces.append(np.array([[0, -1, 0, 0], [-np.sin(span), np.cos(span), 0, 0]]))
    inner = np.array([np.cos(span / 2), np.sin(span / 2), 0.0]) * r / 3
    return np.vstack(halfspaces), inner


def _cut_halfspaces(style: str, spec: PieceSpec, rng: np.random.Generator) -> np.ndarray:
    lo, hi = spec.cut_offset
    if style == "random":
        # 正多面体（6・8・12 面）の面の向きをランダムに回し、1 枚ずつ揺らす。
        # 完全にランダムな向きだと切れていない側が残り、大きすぎる片になりやすい
        normals = _POLY_NORMALS[int(rng.integers(len(_POLY_NORMALS)))] @ _random_rotation(rng).T
        normals = normals + rng.normal(scale=0.25, size=normals.shape)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    elif style == "dice":
        rot = _random_rotation(rng)
        normals = np.vstack([rot.T, -rot.T])
    else:
        raise ValueError(f"unknown cut style: {style}")
    offsets = rng.uniform(lo, hi, size=len(normals))
    return np.column_stack([normals, -offsets])


def min_width(vertices: np.ndarray, n_dirs: int = 400) -> float:
    """凸多面体の最小幅（全方向で測った厚みの最小値）の近似."""
    hull = ConvexHull(vertices)
    sampled = _random_unit_vectors(np.random.default_rng(0), n_dirs)
    dirs = np.vstack([hull.equations[:, :3], sampled])
    proj = vertices @ dirs.T
    return float((proj.max(axis=0) - proj.min(axis=0)).min())


def support_margin(mesh: trimesh.Trimesh, tol: float = 0.0005) -> float:
    """置いた向き（z が上）で、重心を真下に下ろした点から接地面の端までの距離 [m].

    接地面は一番低い頂点から ``tol`` 以内の頂点の凸包。重心が接地面の外なら負の値。
    """
    v = mesh.vertices
    support = v[v[:, 2] < v[:, 2].min() + tol][:, :2]
    if len(support) < 3:
        return -np.inf
    try:
        hull = ConvexHull(support)
    except Exception:  # noqa: BLE001  一直線に並んだ（接地面が線）
        return -np.inf
    # hull.equations は外向き法線 n と d で n·x + d <= 0 が内側。距離は -(n·c + d)
    return float(-(hull.equations[:, :2] @ mesh.center_mass[:2] + hull.equations[:, 2]).max())


def _rest_pose(mesh: trimesh.Trimesh, min_margin: float = 0.0) -> trimesh.Trimesh | None:
    """安定姿勢のうち、外接箱で鉛直が一番薄くなる向きを選んで返す（なければ None）."""
    transforms, probs = trimesh.poses.compute_stable_poses(mesh, n_samples=1)
    for T in transforms[np.argsort(-probs)][:4]:
        posed = mesh.copy()
        posed.apply_transform(T)
        if support_margin(posed) < min_margin:
            continue
        # 鉛直軸まわりに回して、上から見た外接長方形を最小にする
        xy = posed.vertices[:, :2]
        best = None
        for th in np.linspace(0, np.pi / 2, 46):
            c, s = np.cos(th), np.sin(th)
            ext = np.ptp(xy @ np.array([[c, -s], [s, c]]), axis=0)
            if best is None or ext.prod() < best[0]:
                best = (ext.prod(), th)
        rot = trimesh.transformations.rotation_matrix(-best[1], [0, 0, 1])
        posed.apply_transform(rot)
        ext = posed.extents
        if ext[2] <= ext[0] + 1e-6 and ext[2] <= ext[1] + 1e-6:
            posed.apply_translation(-posed.center_mass)
            return posed
    return None


def make_piece(
    food: str, style: str, rng: np.random.Generator, spec: PieceSpec | None = None, tries: int = 50
) -> trimesh.Trimesh:
    """条件を満たす小片を 1 つ作る（重心が原点、安定姿勢で z が上）."""
    spec = spec if spec is not None else PieceSpec()
    for _ in range(tries):
        if style == "rounded":
            halfspaces, center = _rounded_halfspaces(rng)
        elif style == "ichou":
            halfspaces, center = _ichou_halfspaces(rng)
        else:
            base, inner = _base_halfspaces(food, rng)
            center = rng.uniform(-inner, inner)
            cuts = _cut_halfspaces(style, spec, rng)
            cuts[:, 3] -= cuts[:, :3] @ center  # 切断面を中心 center のまわりに置く
            halfspaces = np.vstack([base, cuts])
        try:
            verts = HalfspaceIntersection(halfspaces, center).intersections
        except Exception:  # noqa: BLE001  中心が内部にない・退化した場合は作り直す
            continue
        mesh = trimesh.convex.convex_hull(verts)
        if not mesh.is_volume:
            continue
        if min_width(mesh.vertices) < spec.min_width or mesh.extents.max() > spec.max_extent:
            continue
        posed = _rest_pose(mesh, spec.min_support_margin)
        if posed is not None and posed.extents.max() <= spec.max_extent:
            return posed
    raise RuntimeError(f"failed to make a valid {food}/{style} piece in {tries} tries")


def piece_info(name: str, food: str, style: str, mesh: trimesh.Trimesh) -> PieceInfo:
    return PieceInfo(
        name=name,
        food=food,
        style=style,
        volume_cm3=round(float(mesh.volume) * 1e6, 3),
        mass_kg=round(float(mesh.volume) * FOOD_DENSITY[food], 5),
        extents_mm=tuple(round(float(e) * 1e3, 1) for e in mesh.extents),
        min_width_mm=round(min_width(mesh.vertices) * 1e3, 1),
        n_vertices=len(mesh.vertices),
    )
