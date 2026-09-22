"""
Face Protection -- reduces the facial-feature warping seen on close-up faces at
strong 3D Strength / Pop-Out Boost (real user report, 2026-09-22).

Why it's needed
----------------
The real depth difference between a nose tip and an eye socket is only a few
centimetres. When a face fills the frame, that tiny relief gets normalised to
use almost the whole 0..1 depth range like anything else in the shot -- and a
strong Divergence or Pop-Out Boost (ADR-196) multiplies whatever sits in front
of the Convergence plane, so this already-exaggerated relief is stretched
again. The nose reads as pulled toward the audience and the eyes look sunken
or warped -- a real, reported artifact, not a bug in Boost itself: Boost is
doing exactly what it's told, the depth map handed to it already overstates a
face's own relief once the face fills the frame.

What this does
---------------
Detects faces in the RGB frame (OpenCV Haar cascade -- the same, already-
bundled detector iw3.face_convergence_estimator uses for face_detect
convergence mode; no model download, no GPU needed for the detector itself),
and for each detected face, pulls that region's depth toward its own median by
`strength` (0 = untouched, 1 = fully flattened to one plane) -- BEFORE Pop-Out
Boost/Max Negative Parallax sees it (see apply_divergence in utils.py), so
whatever those settings then do to the frame, the face's own internal relief
is smaller to begin with, and the nose/eye distortion has less to work with.

The mask feathers out from each detected box (proportional to the box's own
size, so a face further from camera gets a proportionally finer edge) -- no
hard rectangle line ever appears in the depth map -- and several faces just
blend via a pointwise maximum of their own masks, so overlapping faces need no
special-casing.

Runs once per frame; no temporal smoothing of its own (matches Edge Fix's own
behaviour) -- Flicker Reduction, if on, still smooths the RESULT same as
always. A frame with no detected face, or no OpenCV, is returned unchanged.
"""
import torch
import torch.nn.functional as F

try:
    import cv2
    _cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    _face_cascade = cv2.CascadeClassifier(_cascade_path)
    FACE_CASCADE_AVAILABLE = not _face_cascade.empty()
except Exception:                                              # noqa: BLE001
    FACE_CASCADE_AVAILABLE = False
    _face_cascade = None


def detect_face_boxes(rgb_chw):
    """rgb_chw: (3, H, W) float tensor, 0..1. Returns a list of (x, y, w, h) integer pixel boxes, or []
    if OpenCV isn't available or nothing was found. Mirrors face_convergence_estimator.py's own detection
    call exactly (same cascade parameters), so the two features agree on what counts as a face."""
    if not FACE_CASCADE_AVAILABLE:
        return []
    frame_rgb = (rgb_chw.permute(1, 2, 0).detach().cpu().float() * 255).clamp(0, 255).byte().numpy()
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return [tuple(int(v) for v in box) for box in faces]


def _feathered_box_mask(H, W, boxes, device):
    """(H, W) float mask: 1.0 inside each detected box, fading smoothly to 0 over a single feather band
    (20% of the box's own longer side, so a small, distant face gets a finer edge than a large, close
    one) -- the mask reaches `feather` pixels beyond the raw box and no further. Several boxes take a
    pointwise maximum, so overlapping faces blend into one region with no seam.

    (Real bug found live in testing: an earlier version both padded the box outward for a hair/chin
    margin AND used that same padding again as the blur radius, so the two compounded and the affected
    region reached 60% further beyond the face than intended. Fixed by feathering the raw box directly,
    with one width, used once.)"""
    mask = torch.zeros(H, W, device=device)
    for (x, y, w, h) in boxes:
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W, x + w), min(H, y + h)
        if x2 <= x1 or y2 <= y1:
            continue
        feather = max(2, int(round(max(w, h) * 0.20)))
        box = torch.zeros(1, 1, H, W, device=device)
        box[0, 0, y1:y2, x1:x2] = 1.0
        box = F.pad(box, (feather, feather, feather, feather), mode="replicate")
        box = F.avg_pool2d(box, kernel_size=feather * 2 + 1, stride=1)
        mask = torch.maximum(mask, box[0, 0])
    return mask


def protect_faces(depth, im, strength):
    """depth: (B, 1, H, W), im: (B, 3, H, W) -- already the shape apply_divergence works with. Returns
    depth with each detected face's own internal relief pulled toward its median by `strength` (0 = no-op,
    returns depth unchanged; 1 = each face fully flattened to one plane). Never raises: a batch with no
    detected faces anywhere, or no OpenCV, returns the original tensor untouched (no wasted clone)."""
    if strength <= 0.0 or not FACE_CASCADE_AVAILABLE:
        return depth
    B, _, H, W = depth.shape
    out = None
    for i in range(B):
        boxes = detect_face_boxes(im[i])
        if not boxes:
            continue
        mask = _feathered_box_mask(H, W, boxes, depth.device).view(1, 1, H, W)
        d = depth[i:i + 1]
        face_depth = d[mask > 0.5]
        if face_depth.numel() == 0:
            continue
        median = face_depth.median()
        flattened = median + (d - median) * (1.0 - strength)
        if out is None:
            out = depth.clone()
        out[i:i + 1] = d * (1 - mask) + flattened * mask
    return out if out is not None else depth
