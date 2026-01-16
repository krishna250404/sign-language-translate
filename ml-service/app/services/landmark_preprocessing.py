import numpy as np

def normalize_landmarks(landmarks):
    """
    landmarks: list of 21 mediapipe landmarks
    returns: np.array shape (63,)
    """

    coords = np.array(
        [[lm.x, lm.y, lm.z] for lm in landmarks],
        dtype=np.float32
    )

    # Use wrist as origin
    coords -= coords[0]

    # Scale normalization
    max_dist = np.max(np.linalg.norm(coords, axis=1))
    if max_dist > 0:
        coords /= max_dist

    return coords.flatten()
