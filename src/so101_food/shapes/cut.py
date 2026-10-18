"""食材の小片（凸多面体）を「切る」ことで作る形状生成器.

元の形（にんじん = 細い円柱、じゃがいも = 楕円体、大根 = 太い円柱）を半空間の集まりで
近似し、ランダムな切断面（半空間）を足して交差を取ると、凸な小片が得られる。
交差の計算は scipy の HalfspaceIntersection で行う（元の形も切断面も凸なので結果も凸）。

切り方:

- ``random``（乱切り）: 小片の中心から 4〜6 方向にランダムな平面で切る
- ``dice``  （角切り）: 直交する 3 組の平面で切る（向きはランダム）

出来た小片は次を満たすものだけを残す（満たさなければ作り直す）。

- 最も薄い方向の厚み（最小幅）が ``min_width`` 以上（薄切りは対象外）
- 最も長い寸法が ``max_extent`` 以下（グリッパに収まる）

最後に、机に置いたときに安定する向き（安定姿勢）に回し、外接箱の一番薄い軸が鉛直
になるようにそろえる（so101-nexus は外接箱の一番薄い軸を上にして置くため）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import trimesh
from scipy.spatial import ConvexHull, HalfspaceIntersection

FOOD_TYPES = ("carrot", "potato", "daikon")
CUT_STYLES = ("random", "dice")
DENSITY = 950.0  # kg/m^3（生野菜はおよそ水と同じ）


@dataclass(frozen=True)
class PieceSpec:
    min_width: float = 0.015  # 最小の厚み [m]
    max_extent: float = 0.035  # 最大の寸法 [m]
    cut_offset: tuple[float, float] = (0.008, 0.016)  # 中心から切断面までの距離 [m]


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
    """元の形を半空間 ``A x + b <= 0`` の集まりで返す（scipy の形式 [A | b]）と内部の点の範囲."""
    if food in ("carrot", "daikon"):
        r = rng.uniform(0.011, 0.016) if food == "carrot" else rng.uniform(0.024, 0.032)
        half_len = 0.06
        ang = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        normals = np.column_stack([np.zeros_like(ang), np.cos(ang), np.sin(ang)])  # 軸は x
        offsets = np.full(len(ang), r)
        normals = np.vstack([normals, [[1, 0, 0], [-1, 0, 0]]])
        offsets = np.concatenate([offsets, [half_len, half_len]])
        inner = np.array([half_len - 0.02, r * 0.4, r * 0.4])
    elif food == "potato":
        axes = np.array(
            [rng.uniform(0.030, 0.040), rng.uniform(0.022, 0.030), rng.uniform(0.020, 0.026)]
        )
        sphere = trimesh.creation.icosphere(subdivisions=2).vertices
        points = sphere * axes  # 楕円体の表面の点
        normals = sphere / axes  # その点での外向き法線（未正規化）
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        offsets = np.einsum("ij,ij->i", normals, points)
        inner = axes * 0.45
    else:
        raise ValueError(f"unknown food: {food}")
    halfspaces = np.column_stack([normals, -offsets])
    return halfspaces, inner


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


def _rest_pose(mesh: trimesh.Trimesh) -> trimesh.Trimesh | None:
    """安定姿勢のうち、外接箱で鉛直が一番薄くなる向きを選んで返す（なければ None）."""
    transforms, probs = trimesh.poses.compute_stable_poses(mesh, n_samples=1)
    for T in transforms[np.argsort(-probs)][:4]:
        posed = mesh.copy()
        posed.apply_transform(T)
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
        base, inner = _base_halfspaces(food, rng)
        center = rng.uniform(-inner, inner)
        cuts = _cut_halfspaces(style, spec, rng)
        cuts[:, 3] -= cuts[:, :3] @ center  # 切断面を中心 center のまわりに置く
        try:
            verts = HalfspaceIntersection(np.vstack([base, cuts]), center).intersections
        except Exception:  # noqa: BLE001  中心が内部にない・退化した場合は作り直す
            continue
        mesh = trimesh.convex.convex_hull(verts)
        if not mesh.is_volume:
            continue
        if min_width(mesh.vertices) < spec.min_width or mesh.extents.max() > spec.max_extent:
            continue
        posed = _rest_pose(mesh)
        if posed is not None and posed.extents.max() <= spec.max_extent:
            return posed
    raise RuntimeError(f"failed to make a valid {food}/{style} piece in {tries} tries")


def piece_info(name: str, food: str, style: str, mesh: trimesh.Trimesh) -> PieceInfo:
    return PieceInfo(
        name=name,
        food=food,
        style=style,
        volume_cm3=round(float(mesh.volume) * 1e6, 3),
        mass_kg=round(float(mesh.volume) * DENSITY, 5),
        extents_mm=tuple(round(float(e) * 1e3, 1) for e in mesh.extents),
        min_width_mm=round(min_width(mesh.vertices) * 1e3, 1),
        n_vertices=len(mesh.vertices),
    )
