import numpy as np, math

def normalize(q):
    q = np.array(q, dtype=float)
    n = np.linalg.norm(q)
    return q / n if n>1e-12 else np.array([1.,0.,0.,0.])

def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)

def quat_mul(a,b):
    w1,x1,y1,z1 = a; w2,x2,y2,z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=float)

def quat_to_euler_zyx(q):
    w,x,y,z = q
    sinr = 2*(w*x + y*z); cosr = 1 - 2*(x*x + y*y)
    roll = math.atan2(sinr, cosr)
    sinp = 2*(w*y - z*x)
    pitch = math.copysign(math.pi/2, sinp) if abs(sinp)>=1 else math.asin(sinp)
    siny = 2*(w*z + x*y); cosy = 1 - 2*(y*y + z*z)
    yaw = math.atan2(siny, cosy)
    return roll, pitch, yaw  # radians

# Parameters from your mapping
mapping = {'xAxis':'roll','yAxis':'yaw','invX':True,'invY':False}

# q0 = baseline quaternion [w,x,y,z] (set from calibration)
# q_in = incoming quaternion [w,x,y,z] from IMU
def apply_calibration_and_map(q_in, q0=None, qg=None):
    q_in = normalize(q_in)
    # 1) bring into baseline frame
    if q0 is not None:
        q_rel = quat_mul(quat_conj(q0), q_in)
    else:
        q_rel = q_in
    # 2) apply optional global rotation (qg) if you have R_global -> qg
    q_out = quat_mul(qg, q_rel) if qg is not None else q_rel
    q_out = normalize(q_out)
    # 3) convert to euler
    roll, pitch, yaw = quat_to_euler_zyx(q_out)
    axis_vals = {'roll': roll, 'pitch': pitch, 'yaw': yaw}
    vx = axis_vals[mapping['xAxis']]
    vy = axis_vals[mapping['yAxis']]
    if mapping['invX']: vx = -vx
    if mapping['invY']: vy = -vy
    # now vx,vy are radians; map to normalized [-1..1] for cursor:
    # simple tanh scaling + sensitivity
    sensitivity = 1.0
    nx = math.tanh(vx * sensitivity)
    ny = math.tanh(vy * sensitivity)
    return nx, ny, (roll,pitch,yaw)
