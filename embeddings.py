"""
Face embedding extraction using InsightFace (ArcFace / ONNX, Python 3.13 compatible).
Downloads ~100 MB of ONNX models on first run to INSIGHTFACE_HOME (default ~/.insightface).
"""
from __future__ import annotations
import fcntl
import os
import numpy as np

_MODEL_ROOT = os.environ.get("INSIGHTFACE_HOME", os.path.expanduser("~/.insightface"))


class FaceEmbedder:
    """
    Wraps InsightFace FaceAnalysis.
    Uses 'buffalo_s' (small) model for fast inference on macOS.
    Switch to 'buffalo_l' for higher accuracy at the cost of speed.
    """

    MODEL_NAME = "buffalo_s"  # buffalo_s | buffalo_m | buffalo_l

    def __init__(self):
        from insightface.app import FaceAnalysis
        os.makedirs(_MODEL_ROOT, exist_ok=True)
        # File lock prevents concurrent processes from corrupting the model download
        lock_path = os.path.join(_MODEL_ROOT, ".download.lock")
        with open(lock_path, "w") as _lf:
            fcntl.flock(_lf, fcntl.LOCK_EX)
            # CoreML EP has a shape-rank mismatch with InsightFace's SCRFD detector — CPU only
            self._app = FaceAnalysis(
                name=self.MODEL_NAME,
                root=_MODEL_ROOT,
                providers=["CPUExecutionProvider"],
            )
            self._app.prepare(ctx_id=0, det_size=(320, 320))

    # ------------------------------------------------------------------

    def get_embedding(self, img_bgr: np.ndarray) -> np.ndarray | None:
        """
        Detect faces in img_bgr and return a normalized 512-dim ArcFace
        embedding for the largest detected face, or None if no face found.
        """
        faces = self._app.get(img_bgr)
        if not faces:
            return None
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        emb  = face.embedding.astype(np.float32)
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    def get_all_embeddings(self, img_bgr: np.ndarray) -> list[np.ndarray]:
        """Return normalized embeddings for ALL detected faces (used in admin panel)."""
        faces = self._app.get(img_bgr)
        result = []
        for f in faces:
            emb  = f.embedding.astype(np.float32)
            norm = np.linalg.norm(emb)
            result.append(emb / norm if norm > 0 else emb)
        return result

    # ------------------------------------------------------------------

    @staticmethod
    def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity (0–1) between two normalized embeddings."""
        return float(np.dot(a, b))

    @staticmethod
    def best_match(
        query: np.ndarray,
        db_embeddings: list[tuple[str, np.ndarray]],
        threshold: float = 0.48,
    ) -> tuple[str | None, float]:
        """
        Compare query against all stored embeddings.  When a user has multiple
        embeddings (different angles/lighting), we take the max similarity across
        all of that user's entries — so more photos always helps.

        Returns (best_user_id, best_similarity); user_id is None if below threshold.
        """
        if not db_embeddings:
            return None, 0.0

        refs = np.stack([e for _, e in db_embeddings])  # (N, 512)
        sims = refs @ query                              # (N,) cosine sims

        # Group by user: take the max sim per user_id
        best_per_user: dict[str, float] = {}
        for (uid, _), sim in zip(db_embeddings, sims.tolist()):
            if uid not in best_per_user or sim > best_per_user[uid]:
                best_per_user[uid] = sim

        best_uid  = max(best_per_user, key=best_per_user.get)
        best_sim  = best_per_user[best_uid]

        if best_sim >= threshold:
            return best_uid, best_sim
        return None, best_sim
