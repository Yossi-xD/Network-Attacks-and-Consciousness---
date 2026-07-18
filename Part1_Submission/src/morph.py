"""Landmark-based face morphing: Delaunay triangulation + piecewise-affine
warp + cross-dissolve, following the classic Beier-Neely / Delaunay morphing
approach used in face-morphing-attack literature (e.g. Ferrara et al., 2014).
"""
import cv2
import numpy as np


def delaunay_triangulation(points, size):
    """Triangulate `points` (Nx2) inside a `size`=(w,h) rectangle.
    Returns a list of triangles as index-triplets into `points`."""
    w, h = size
    points = points.copy()
    points[:, 0] = np.clip(points[:, 0], 0, w - 1)
    points[:, 1] = np.clip(points[:, 1], 0, h - 1)
    subdiv = cv2.Subdiv2D((0, 0, w, h))
    for p in points:
        subdiv.insert((float(p[0]), float(p[1])))

    # map each (x, y) vertex back to its point index
    point_index = {(round(p[0], 1), round(p[1], 1)): i for i, p in enumerate(points)}

    triangles = []
    for t in subdiv.getTriangleList():
        tri_pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        if not all(0 <= x < w and 0 <= y < h for x, y in tri_pts):
            continue
        idx = []
        for x, y in tri_pts:
            key = (round(x, 1), round(y, 1))
            if key not in point_index:
                # snap to nearest point (numerical rounding from Subdiv2D)
                dists = np.hypot(points[:, 0] - x, points[:, 1] - y)
                idx.append(int(np.argmin(dists)))
            else:
                idx.append(point_index[key])
        if len(set(idx)) == 3:
            triangles.append(tuple(idx))
    return triangles


def _warp_triangle(src_img, dst_img, tri_src, tri_dst):
    """Affine-warp the triangular patch `tri_src` from src_img into the
    triangular region `tri_dst` of dst_img (in place, additive via mask)."""
    r_src = cv2.boundingRect(np.float32([tri_src]))
    r_dst = cv2.boundingRect(np.float32([tri_dst]))

    tri_src_rect = [(p[0] - r_src[0], p[1] - r_src[1]) for p in tri_src]
    tri_dst_rect = [(p[0] - r_dst[0], p[1] - r_dst[1]) for p in tri_dst]

    src_patch = src_img[r_src[1]:r_src[1] + r_src[3], r_src[0]:r_src[0] + r_src[2]]
    if src_patch.size == 0 or r_dst[2] <= 0 or r_dst[3] <= 0:
        return

    mat = cv2.getAffineTransform(np.float32(tri_src_rect), np.float32(tri_dst_rect))
    warped = cv2.warpAffine(src_patch, mat, (r_dst[2], r_dst[3]), None,
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    mask = np.zeros((r_dst[3], r_dst[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(tri_dst_rect), (1.0, 1.0, 1.0), cv2.LINE_AA)

    # clip the destination rect to the canvas -- triangles touching the
    # image border can have a bounding rect that pokes 1px past the edge
    H, W = dst_img.shape[:2]
    x0, y0, w0, h0 = r_dst
    x1c, y1c = max(x0, 0), max(y0, 0)
    x2c, y2c = min(x0 + w0, W), min(y0 + h0, H)
    if x2c <= x1c or y2c <= y1c:
        return
    ox0, oy0 = x1c - x0, y1c - y0
    ox1, oy1 = ox0 + (x2c - x1c), oy0 + (y2c - y1c)

    warped_c = warped[oy0:oy1, ox0:ox1]
    mask_c = mask[oy0:oy1, ox0:ox1]
    dst_slice = dst_img[y1c:y2c, x1c:x2c]
    dst_slice[:] = dst_slice * (1 - mask_c) + warped_c * mask_c


def morph_images(img1, pts1, img2, pts2, alpha, triangles=None):
    """Produce the alpha-blended morph of img1/img2 given matching landmark
    sets pts1/pts2 (Nx2, same ordering/count). alpha=0 -> img1, alpha=1 -> img2.
    `triangles` can be precomputed (via delaunay_triangulation on the averaged
    landmarks) and reused across multiple alpha values for speed/consistency.
    """
    h, w = img1.shape[:2]
    pts1 = np.asarray(pts1, dtype=np.float64)
    pts2 = np.asarray(pts2, dtype=np.float64)
    pts_morph = (1 - alpha) * pts1 + alpha * pts2

    if triangles is None:
        triangles = delaunay_triangulation(pts_morph, (w, h))

    img1f = img1.astype(np.float32)
    img2f = img2.astype(np.float32)
    warped1 = np.zeros_like(img1f)
    warped2 = np.zeros_like(img2f)

    for (i, j, k) in triangles:
        tri1 = [tuple(pts1[i]), tuple(pts1[j]), tuple(pts1[k])]
        tri2 = [tuple(pts2[i]), tuple(pts2[j]), tuple(pts2[k])]
        tri_m = [tuple(pts_morph[i]), tuple(pts_morph[j]), tuple(pts_morph[k])]
        _warp_triangle(img1f, warped1, tri1, tri_m)
        _warp_triangle(img2f, warped2, tri2, tri_m)

    morphed = (1 - alpha) * warped1 + alpha * warped2
    return np.clip(morphed, 0, 255).astype(np.uint8), triangles
