
# [FIX] Helpers defensivos para extrair atributos comuns de robôs/obstáculos
def _get_radius(obj, default=DEFAULT_RADIUS):
    r = getattr(obj, 'radius', None)
    if r is None:
        r = getattr(obj, 'r', None)
    if r is None:
        return float(default)
    return float(np.asarray(r).flatten()[0])


def _get_velocity(obj):
    """Tenta extrair a velocidade (vx, vy) de um objeto do irsim."""
    for attr in ('velocity', 'vel'):
        v = getattr(obj, attr, None)
        if v is not None:
            arr = np.asarray(v).flatten()
            if arr.size >= 2:
                return arr[0:2].astype(np.float64)
    s = np.asarray(getattr(obj, 'state', np.zeros(2))).flatten()
    if s.size >= 4:
        return s[2:4].astype(np.float64)
    return np.zeros(2, dtype=np.float64)


def _get_position(obj):
    s = np.asarray(getattr(obj, 'state', np.zeros(2))).flatten()
    return s[0:2].astype(np.float64)


def _get_goal(obj):
    g = getattr(obj, 'goal', None)
    if g is None:
        return None
    arr = np.asarray(g).flatten()
    if arr.size < 2:
        return None
    return arr[0:2].astype(np.float64)


def _pack_obstacles(obstacles_list):
    """Converte lista de dicts em arrays NumPy pré-alocados para o PyORCA."""
    n = len(obstacles_list)
    if n == 0:
        return (np.empty((0, 2), dtype=np.float64),
                np.empty((0, 2), dtype=np.float64),
                np.empty((0,),   dtype=np.float64))
    obs_pos = np.empty((n, 2), dtype=np.float64)
    obs_v   = np.empty((n, 2), dtype=np.float64)
    obs_r   = np.empty((n,),   dtype=np.float64)
    for i, o in enumerate(obstacles_list):
        obs_pos[i, 0], obs_pos[i, 1] = o['pos']
        obs_v[i, 0],   obs_v[i, 1]   = o['v']
        obs_r[i] = o['radius']
    return obs_pos, obs_v, obs_r


def _check_collision(robot_list):
    """[FIX] Checagem manual de colisão entre robôs (substitui env.done())."""
    for i, r1 in enumerate(robot_list):
        p1 = _get_position(r1)
        rad1 = _get_radius(r1)
        for r2 in robot_list[i + 1:]:
            p2 = _get_position(r2)
            rad2 = _get_radius(r2)
            if np.linalg.norm(p1 - p2) < (rad1 + rad2):
                return True
    return False