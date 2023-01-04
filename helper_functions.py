import numpy as np
import matplotlib.pyplot as plt
#import json
#import toml
from mpl_toolkits.mplot3d import axes3d
import scipy as sci
#import cv2


def projectPoint(point, R, t, K):
    # 3D -> 2D projection from here https://kornia.readthedocs.io/en/v0.1.2/pinhole.html
    point_h = np.ones(4)
    point_h[:3] = point
    G = np.zeros((3,4))
    G[:,:3] = R
    G[:,3] = t.T
    uv_vec = K @ G[:3, :] @ point_h.T
    return uv_vec.T[:2] / uv_vec.T[2], uv_vec.T[2]

def unit_axis_angle(a, b):
    an = np.sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])
    bn = np.sqrt(b[0]*b[0] + b[1]*b[1] + b[2]*b[2])
    ax, ay, az = a[0]/an, a[1]/an, a[2]/an
    bx, by, bz = b[0]/bn, b[1]/bn, b[2]/bn
    nx, ny, nz = ay*bz-az*by, az*bx-ax*bz, ax*by-ay*bx
    nn = np.sqrt(nx*nx + ny*ny + nz*nz)
    return (nx/nn, ny/nn, nz/nn), acos(ax*bx + ay*by + az*bz)

def rotation_matrix(axis, angle):
    ax, ay, az = axis[0], axis[1], axis[2]
    s = np.sin(angle)
    c = np.cos(angle)
    u = 1 - c
    return np.array( [[ ax*ax*u + c,    ax*ay*u - az*s, ax*az*u + ay*s ], [ ay*ax*u + az*s, ay*ay*u + c,    ay*az*u - ax*s ],[ az*ax*u - ay*s, az*ay*u + ax*s, az*az*u + c    ]])

def fit_ellipse(x, y):
    """
    Fit the coefficients a,b,c,d,e,f, representing an ellipse described by
    the formula F(x,y) = ax^2 + bxy + cy^2 + dx + ey + f = 0 to the provided
    arrays of data points x=[x1, x2, ..., xn] and y=[y1, y2, ..., yn].

    Based on the algorithm of Halir and Flusser, "Numerically stable direct
    least squares fitting of ellipses'.
    from https://scipython.com/blog/direct-linear-least-squares-fitting-of-an-ellipse/
    http://autotrace.sourceforge.net/WSCG98.pdf
    """

    D1 = np.vstack([x ** 2, x * y, y ** 2]).T
    D2 = np.vstack([x, y, np.ones(len(x))]).T
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    T = -np.linalg.inv(S3) @ S2.T
    M = S1 + S2 @ T
    C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
    M = np.linalg.inv(C) @ M
    eigval, eigvec = np.linalg.eig(M)
    con = 4 * eigvec[0] * eigvec[2] - eigvec[1] ** 2
    ak = eigvec[:, np.nonzero(con > 0)[0]]
    return np.concatenate((ak, T @ ak)).ravel()


def transform_ellipse(coeffs):
    """

    Convert the cartesian conic coefficients, (a, b, c, d, e, f), to the
    ellipse parameters, where F(x, y) = ax^2 + bxy + cy^2 + dx + ey + f = 0.
    The returned parameters are x0, y0, ap, bp, e, phi, where (x0, y0) is the
    ellipse centre; (ap, bp) are the semi-major and semi-minor axes,
    respectively; e is the eccentricity; and phi is the rotation of the semi-
    major axis from the x-axis.
    from https://scipython.com/blog/direct-linear-least-squares-fitting-of-an-ellipse/
    http://autotrace.sourceforge.net/WSCG98.pdf
    """

    # We use the formulas from https://mathworld.wolfram.com/Ellipse.html
    # which assumes a cartesian form ax^2 + 2bxy + cy^2 + 2dx + 2fy + g = 0.
    # Therefore, rename and scale b, d and f appropriately.
    a = coeffs[0]
    b = coeffs[1] / 2
    c = coeffs[2]
    d = coeffs[3] / 2
    f = coeffs[4] / 2
    g = coeffs[5]

    den = b ** 2 - a * c
    if den > 0:
        raise ValueError('coeffs do not represent an ellipse: b^2 - 4ac must'
                         ' be negative!')

    # The location of the ellipse centre.
    x0, y0 = (c * d - b * f) / den, (a * f - b * d) / den

    num = 2 * (a * f ** 2 + c * d ** 2 + g * b ** 2 - 2 * b * d * f - a * c * g)
    fac = np.sqrt((a - c) ** 2 + 4 * b ** 2)
    # The semi-major and semi-minor axis lengths (these are not sorted).
    ap = np.sqrt(num / den / (fac - a - c))
    bp = np.sqrt(num / den / (-fac - a - c))

    # Sort the semi-major and semi-minor axis lengths but keep track of
    # the original relative magnitudes of width and height.
    width_gt_height = True
    if ap < bp:
        width_gt_height = False
        ap, bp = bp, ap

    # The eccentricity.
    r = (bp / ap) ** 2
    if r > 1:
        r = 1 / r
    e = np.sqrt(1 - r)

    # The angle of anticlockwise rotation of the major-axis from x-axis.
    if b == 0:
        phi = 0 if a < c else np.pi / 2
    else:
        phi = np.arctan((2. * b) / (a - c)) / 2
        if a > c:
            phi += np.pi / 2
    if not width_gt_height:
        # Ensure that phi is the angle to rotate to the semi-major axis.
        phi += np.pi / 2
    phi = phi % np.pi

    return x0, y0, ap, bp, e, phi


def plot_implicit(fn, Q, bbox=(-2.5, 2.5)):
    ''' create a plot of an implicit function
    fn  ...implicit function (plot where fn==0)
    bbox ..the x,y,and z limits of plotted interval'''
    xmin, xmax, ymin, ymax, zmin, zmax = bbox * 3
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    A = np.linspace(xmin, xmax, 100)  # resolution of the contour
    B = np.linspace(xmin, xmax, 15)  # number of slices
    A1, A2 = np.meshgrid(A, A)  # grid on which the contour is plotted

    for z in B:  # plot contours in the XY plane
        X, Y = A1, A2
        Z = fn(Q, X, Y, z)
        cset = ax.contour(X, Y, Z + z, [z], zdir='z')
        # [z] defines the only level to plot for this contour for this value of z

    for y in B:  # plot contours in the XZ plane
        X, Z = A1, A2
        Y = fn(Q, X, y, Z)
        cset = ax.contour(X, Y + y, Z, [y], zdir='y')

    for x in B:  # plot contours in the YZ plane
        Y, Z = A1, A2
        X = fn(Q, x, Y, Z)
        cset = ax.contour(X + x, Y, Z, [x], zdir='x')

    # must set plot limits because the contour will likely extend
    # way beyond the displayed level.  Otherwise matplotlib extends the plot limits
    # to encompass all values in the contour.
    ax.set_zlim3d(zmin, zmax)
    ax.set_xlim3d(xmin, xmax)
    ax.set_ylim3d(ymin, ymax)

    plt.show()


def dot(A, B):
    return np.multiply(A, B).sum(axis=0)


def norm(A):
    return np.linalg.norm(A)


def cross(A, B):
    return np.cross(A, B)


def calc_G(A, B):
    G = np.array([[dot(A, B), -norm(cross(A, B)), 0], [norm(cross(A, B)), dot(A, B), 0], [0, 0, 1]])
    return G


def calc_F(A, B):
    F = np.array([A, (B - dot(A, B) * A) / norm(B - dot(A, B) * A), cross(B, A)])
    return F


def calc_R(A, B):
    G = calc_G(A, B)
    F = calc_F(A, B)
    F = F.T
    R = F @ G @ np.linalg.inv(F)

    return R


def calc_A(ellipse, camera_matrice):
    tx, ty = ellipse[3], ellipse[4]
    a, b = ellipse[5], ellipse[6]
    phi = ellipse[7]

    fx = camera_matrice[0, 0]
    fy = camera_matrice[1, 1]
    cx = camera_matrice[2, 0]
    cy = camera_matrice[2, 1]

    # scale ellipse to unit

    tx_hat = 1 / fx * (tx - cx)
    ty_hat = 1 / fy * (ty - cy)
    a_hat = a / fx
    b_hat = b / fy

    S = np.array([[np.cos(phi), -np.sin(phi), -tx_hat * np.cos(phi) + ty_hat * np.sin(phi)],
                  [np.sin(phi), np.cos(phi), -tx_hat * np.sin(phi) - ty_hat * np.cos(phi)], [0, 0, 1]])
    H = np.array([[1 / a_hat ** 2, 0, 0], [0, 1 / b_hat ** 2, 0], [0, 0, -1]])
    A = S.T @ H @ S
    return A


def calc_R_t_stereo(left_cam_origin, left_cam_dir_world, right_cam_origin, right_cam_dir_world,
                    world_cords=np.array([0, 0, -1])):
    origin_world = np.array([0, 0, 0])
    origin_world = origin_world.T
    left_cam_origin = left_cam_origin.T
    right_cam_origin = right_cam_origin.T
    t_wcl = left_cam_origin - origin_world
    t_wcr = right_cam_origin - origin_world
    y_camera_left = left_cam_dir_world - left_cam_origin
    y_camera_right = right_cam_dir_world - right_cam_origin
    y_cwl = y_camera_left - t_wcl
    y_cwr = y_camera_right - t_wcr
    y_cwl_unit = y_cwl / norm(y_cwl)
    y_cwr_unit = y_cwr / norm(y_cwr)
    R_cwl = calc_R(world_cords, y_cwl_unit)
    R_cwr = calc_R(world_cords, y_cwr_unit)

    R_lr = R_cwr@R_cwl.T
    R_rl = R_cwl@R_cwr.T

    t_lr = t_wcr - R_cwr@R_cwl.T@t_wcl
    t_rl = t_wcl - R_cwl@R_cwr.T@t_wcr
    return R_lr, R_rl, t_lr, t_rl, R_cwl, t_wcl, R_cwr, t_wcr


def eval_conic_mat(Q, x, y, z):
    u = np.array([x, y, z])
    res = u @ Q @ u.T
    return res


def construct_ellipse(ellipse, camera_matrice):  # x0, y0, ap, bp, e, phi
    x0, y0 = ellipse[0], ellipse[1]
    a, b = ellipse[2], ellipse[3]
    phi = ellipse[5]
    fx = camera_matrice[0, 0]
    fy = camera_matrice[1, 1]
    cx = camera_matrice[0, 2]
    cy = camera_matrice[1, 2]
    #x0 = x0 - cx
    #y0 = y0 - cy
    A = a ** 2 * np.sin(phi) * np.sin(phi) + b ** 2 * np.cos(phi) * np.cos(phi)
    B = 2 * (b ** 2 - a ** 2) * np.sin(phi) * np.cos(phi)
    C = a ** 2 * np.cos(phi) * np.cos(phi) + b ** 2 * np.sin(phi) * np.sin(phi)
    D = -2 * A * x0 - B * y0
    E = -B * x0 - 2 * C * y0
    F = A * x0 ** 2 + B * x0 * y0 + C * y0 ** 2 - a ** 2 * b ** 2
    A_mat = np.array([[A, 0.5 * B, 0.5 * D], [0.5 * B, C, 0.5 * E], [0.5 * D, 0.5 * E, F]])

    return A_mat

def construct_ellipse_real(ellipse, camera_matrice):  # x0, y0, ap, bp, e, phi
    x0, y0 = ellipse[0], ellipse[1]
    a, b = ellipse[2], ellipse[3]
    phi = ellipse[5]
    fx = camera_matrice[0, 0]
    fy = camera_matrice[1, 1]
    cx = camera_matrice[0, 2]
    cy = camera_matrice[1, 2]
    x0 = 1/fx*(x0 - cx)
    y0 = 1/fy*(y0 - cy)
    a = a/fx
    b = b /fy
    A = a ** 2 * np.sin(phi) * np.sin(phi) + b ** 2 * np.cos(phi) * np.cos(phi)
    B = 2 * (b ** 2 - a ** 2) * np.sin(phi) * np.cos(phi)
    C = a ** 2 * np.cos(phi) * np.cos(phi) + b ** 2 * np.sin(phi) * np.sin(phi)
    D = -2 * A * x0 - B * y0
    E = -B * x0 - 2 * C * y0
    F = A * x0 ** 2 + B * x0 * y0 + C * y0 ** 2 - a ** 2 * b ** 2
    A_mat = np.array([[A, 0.5 * B, 0.5 * D], [0.5 * B, C, 0.5 * E], [0.5 * D, 0.5 * E, F]])

    return A_mat


def calc_A(ellipse, camera_matrice):
    x0, y0 = ellipse[0], ellipse[1]
    a, b = ellipse[2], ellipse[3]
    phi = ellipse[5]

    fx = camera_matrice[0, 0]
    fy = camera_matrice[1, 1]
    cx = camera_matrice[2, 0]
    cy = camera_matrice[2, 1]

    # scale ellipse to unit

    tx_hat = 1 / fx * (x0 - cx)
    ty_hat = 1 / fy * (y0 - cy)
    a_hat = a / fx
    b_hat = b / fy

    S = np.array([[np.cos(phi), -np.sin(phi), -tx_hat * np.cos(phi) + ty_hat * np.sin(phi)],
                  [np.sin(phi), np.cos(phi), -tx_hat * np.sin(phi) - ty_hat * np.cos(phi)], [0, 0, 1]])
    H = np.array([[1 / a_hat ** 2, 0, 0], [0, 1 / b_hat ** 2, 0], [0, 0, -1]])
    A = S.T @ H @ S
    return A



def scaleEllipse(A, alpha):
    T = np.diag([alpha, alpha, 1])
    A_scaled = T * A * T

    return A_scaled


def construct_Conic(a, b):
    return np.array([[1 / a ** 2, 0, 0], [0, 1 / b ** 2, 0], [0, 0, -2]])


def eval_mat(x, Mat):
    return x.T @ Mat @ x


import scipy.optimize
from functools import partial


def project(point, K, G):
    # 3D -> 2D projection from here https://kornia.readthedocs.io/en/v0.1.2/pinhole.html
    point_h = np.ones(4)
    point_h[:3] = point
    uv_vec = K @ G[:3, :] @ point_h.T
    return uv_vec[:2] / uv_vec[2], uv_vec[2]


def transform_camera_to_world(K, R, t, p_image):
    return np.linalg.inv(R) @ (np.linalg.inv(K) @ p_image.T - t)


def unproject(point, K, R, t, d):
    # https://math.stackexchange.com/questions/4382437/back-projecting-a-2d-pixel-from-an-image-to-its-corresponding-3d-point
    p = np.array([point[0], point[1], 1]).T
    p.shape = (3, 1)
    t.shape = (3, 1)
    pw = np.linalg.inv(R) @ ((np.linalg.inv(K) @ p * d) - t)
    return pw


def plot_implicite(Mat, x_win=[0, 5], y_win=[0, 5]):
    xs = []
    ys = []

    def eval_mat_loc(x, y):
        x_vec = np.array([x, y, 1])
        return x_vec.T @ Mat @ x_vec

    for x in np.linspace(*x_win, num=2000):
        try:
            # A more efficient technique would use the last-found-y-value as a
            # starting point
            y = scipy.optimize.brentq(partial(eval_mat_loc, x), *y_win, xtol=0.5)
        except ValueError:
            # Should we not be able to find a solution in this window.
            pass
        else:
            xs.append(x)
            ys.append(y)

    return xs, ys


def switch_y2z(vec):
    ret_vec = np.zeros(3)
    ret_vec[0] = vec[2]
    ret_vec[1] = vec[0]
    ret_vec[2] = -vec[1]
    return ret_vec