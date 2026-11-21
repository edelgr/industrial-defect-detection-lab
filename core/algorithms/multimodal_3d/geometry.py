"""Group G — educational 3D baseline: registration + geometric deviation to normal references.

This is an IDD Lab EDUCATIONAL BASELINE (not a reproduction of a published deep 3D detector).

Pipeline
1. voxel down-sampling of reference (normal) and test clouds;
2. coarse alignment: centroid + PCA-scale normalisation, then FPFH features + RANSAC (Rusu et al.
   2009; Open3D) — optional;
3. fine alignment: ICP point-to-plane (Chen & Medioni 1992) or point-to-point (Besl & McKay 1992);
4. deviation of every test point to the reference: nearest-neighbour distance (point-to-point) or
   |n·(p−q)| distance to the reference tangent plane (point-to-surface approximation);
5. score = high percentile of the per-point deviation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from core.algorithms.base import ParamSpec
from core.algorithms.torch_common import have


@dataclass
class PointCloudResult:
    points: np.ndarray            # test points (down-sampled) in the REFERENCE frame after alignment
    deviation: np.ndarray         # per-point raw deviation (same units as the input clouds)
    score: float
    transform: np.ndarray         # 4x4 test->reference
    fitness: float
    inlier_rmse: float
    latency_ms: float
    reference_points: np.ndarray | None = None
    steps: dict = field(default_factory=dict)
    score_semantics: str = ""


class PointCloudDeviation:
    method_id = "pc_deviation"
    display_name = "Registration + geometric deviation (educational baseline)"
    requires_fit = True

    @classmethod
    def param_specs(cls):
        return [
            ParamSpec("voxel", "float", 0.0, "Voxel size (0 = auto: 1/150 of the bounding-box diagonal)", 0.0, 100.0, 0.05),
            ParamSpec("coarse", "bool", True, "Coarse alignment (FPFH + RANSAC)"),
            ParamSpec("icp", "choice", "point_to_plane", "Fine alignment (ICP)", choices=["point_to_plane", "point_to_point", "none"]),
            ParamSpec("distance", "choice", "point_to_surface", "Deviation measure", choices=["point_to_surface", "point_to_point"]),
            ParamSpec("percentile", "float", 99.0, "Score percentile", 50.0, 100.0, 0.5),
        ]

    @classmethod
    def availability(cls):
        return have("open3d", "scipy")

    def __init__(self, **params):
        self.params = {p.name: p.default for p in self.param_specs()}
        self.params.update(params)
        self.fitted = False
        self.reference = None

    def model_version(self) -> str:
        import open3d as o3d
        return f"IDD Lab educational baseline · Open3D {o3d.__version__}"

    # ------------------------------------------------------------------ utils
    def _voxel(self, pts: np.ndarray) -> float:
        v = float(self.params["voxel"])
        if v > 0:
            return v
        diag = float(np.linalg.norm(pts.max(0) - pts.min(0)))
        return max(diag / 150.0, 1e-6)

    @staticmethod
    def _pcd(pts):
        import open3d as o3d
        p = o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64))
        return p

    def _prep(self, pts, voxel):
        import open3d as o3d
        p = self._pcd(pts).voxel_down_sample(voxel)
        p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3, max_nn=30))
        return p

    def fit(self, clouds: list[np.ndarray], progress=None, cancel=None) -> None:
        """Merge normal reference clouds (first one defines the reference frame)."""
        import open3d as o3d
        base = np.asarray(clouds[0])
        self.vox = self._voxel(base)
        ref = self._prep(base, self.vox)
        for k, c in enumerate(clouds[1:], 1):
            res = self._align(self._prep(c, self.vox), ref)
            ref += self._prep(c, self.vox).transform(res[0])
            if progress:
                progress(k / max(1, len(clouds) - 1), f"merging reference {k + 1}/{len(clouds)}")
        ref = ref.voxel_down_sample(self.vox)
        ref.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.vox * 3, max_nn=30))
        self.reference = ref
        self.fitted = True

    def _align(self, src, tgt):
        import open3d as o3d
        reg = o3d.pipelines.registration
        T = np.eye(4)
        steps = {}
        # centroid pre-alignment
        T[:3, 3] = np.asarray(tgt.get_center()) - np.asarray(src.get_center())
        if self.params["coarse"]:
            def fpfh(p):
                return reg.compute_fpfh_feature(p, o3d.geometry.KDTreeSearchParamHybrid(radius=self.vox * 5, max_nn=100))
            src_t = o3d.geometry.PointCloud(src).transform(T)
            r = reg.registration_ransac_based_on_feature_matching(
                src_t, tgt, fpfh(src_t), fpfh(tgt), True, self.vox * 1.5,
                reg.TransformationEstimationPointToPoint(False), 3,
                [reg.CorrespondenceCheckerBasedOnEdgeLength(0.9), reg.CorrespondenceCheckerBasedOnDistance(self.vox * 1.5)],
                reg.RANSACConvergenceCriteria(100000, 0.999))
            if r.fitness > 0:
                T = r.transformation @ T
            steps["coarse_fitness"] = float(r.fitness)
        if self.params["icp"] != "none":
            crit = reg.ICPConvergenceCriteria(max_iteration=60)
            # point-to-point first (robust basin of convergence) ...
            r = reg.registration_icp(src, tgt, self.vox * 2, T, reg.TransformationEstimationPointToPoint(), crit)
            best, used = r, "point_to_point"
            # ... then optional point-to-plane refinement, kept only if it does not reduce the overlap
            if self.params["icp"] == "point_to_plane" and r.fitness > 0:
                r2 = reg.registration_icp(src, tgt, self.vox * 2, r.transformation,
                                          reg.TransformationEstimationPointToPlane(), crit)
                if r2.fitness >= r.fitness * 0.98 and r2.fitness > 0:
                    best, used = r2, "point_to_plane"
                else:
                    steps["note"] = "point-to-plane refinement diverged; kept point-to-point result"
            T = best.transformation
            steps.update(icp_fitness=float(best.fitness), icp_rmse=float(best.inlier_rmse), icp_used=used)
            steps["registration_ok"] = best.fitness >= 0.3
            return T, float(best.fitness), float(best.inlier_rmse), steps
        e = reg.evaluate_registration(src, tgt, self.vox * 2, T)
        steps["registration_ok"] = e.fitness >= 0.3
        return T, float(e.fitness), float(e.inlier_rmse), steps

    def predict(self, points: np.ndarray) -> PointCloudResult:
        from scipy.spatial import cKDTree
        if not self.fitted:
            raise RuntimeError("fit() on normal reference clouds first")
        t0 = time.perf_counter()
        src = self._prep(points, self.vox)
        T, fit, rmse, steps = self._align(src, self.reference)
        src.transform(T)
        P = np.asarray(src.points)
        R = np.asarray(self.reference.points)
        tree = cKDTree(R)
        d, j = tree.query(P)
        if self.params["distance"] == "point_to_surface":
            N = np.asarray(self.reference.normals)
            d = np.abs(np.einsum("ij,ij->i", P - R[j], N[j]))
        score = float(np.percentile(d, float(self.params["percentile"])))
        return PointCloudResult(P, d.astype(np.float32), score, T, fit, rmse, (time.perf_counter() - t0) * 1000,
                                reference_points=R, steps={**steps, "voxel": self.vox, "n_test": len(P), "n_ref": len(R)},
                                score_semantics=f"per-point {self.params['distance'].replace('_', '-')} distance after "
                                                f"alignment (input units); score = p{self.params['percentile']:g}")

    def labels_on(self, result: PointCloudResult, full_points: np.ndarray, full_labels: np.ndarray) -> np.ndarray:
        """Transfer GT labels from the original test cloud to the down-sampled, aligned points."""
        from scipy.spatial import cKDTree
        Tinv = np.linalg.inv(result.transform)
        back = (np.c_[result.points, np.ones(len(result.points))] @ Tinv.T)[:, :3]
        _, j = cKDTree(full_points).query(back)
        return full_labels[j]
