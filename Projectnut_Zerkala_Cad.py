"""
Zerkala v2.3 — Инструмент проектирования зеркал
Авторы команда НЕЧТО
"""

import os
import numpy as np
from numpy import sqrt, pi, tan, arange
import warnings

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

from stl import mesh
import ezdxf

# ---------------------------------------------------------------------------
# Вспомогательные функции — общие
# ---------------------------------------------------------------------------

def deg2rad(x=0.0):
    return x * pi / 180.0

def ctg(x):
    return 1.0 / tan(x)

def dist(p1=(0.0, 0.0), p2=(0.0, 0.0)):
    return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# ---------------------------------------------------------------------------
# Гиперболоид
# ---------------------------------------------------------------------------

def signal(a, b, t, ei):
    """Профиль гиперболоида: y = a·√(b²+t²)/b + ei"""
    return (a * sqrt(b**2 + t**2)) / b + ei


def compute_hyp_params(a_val, b_val, w_val, ra_val, F1=(0.0, 0.0)):
    """Пересчёт геометрических параметров гиперболоида по значениям слайдеров."""
    cr  = (w_val, ctg(deg2rad(a_val / 2)) * w_val)
    F2  = (0.0, cr[1] - tan(deg2rad(b_val / 2)) * w_val)
    a   = (dist(F1, cr) - dist(F2, cr)) / 2.0
    ei  = F2[1] / 2.0
    b2  = ei**2 - a**2
    b   = sqrt(max(b2, 1e-12))
    ra  = min(max(ra_val, -a_val / 2.0), a_val / 2.0)
    rx  = tan(deg2rad(ra)) * cr[1]
    ry  = signal(a, b, rx, ei)
    return dict(cr=cr, F1=F1, F2=F2, a=a, ei=ei, b=b, ra=ra, rx=rx, ry=ry)

# ---------------------------------------------------------------------------
# Эквиугловое зеркало  (Stürzl et al., ECCV 2004)
# ---------------------------------------------------------------------------

def equiangular_theta_max_valid(alpha, gamma_s_deg):
    """
    Максимальный допустимый θmax (в градусах), при котором
    cos(θ/k + γS) > 0 для всех θ ∈ [0, θmax].
    """
    k   = 2.0 / (1.0 + alpha)
    lim = k * (90.0 - gamma_s_deg) - 0.05
    return max(lim, 0.1)


def compute_equiangular(alpha, gamma_s_deg, R_max, theta_max_deg, N=500):
    """
    Профиль эквиуглового зеркала (Stürzl et al., ECCV 2004).
    Возвращает (x_vals, z_vals, theta_vals, r_vals, K, k, gamma_s_rad,
                theta_max_used_deg).
    """
    k  = 2.0 / (1.0 + alpha)
    gs = deg2rad(gamma_s_deg)

    theta_max_lim = equiangular_theta_max_valid(alpha, gamma_s_deg)
    theta_max_deg = min(theta_max_deg, theta_max_lim)
    theta_max     = deg2rad(max(theta_max_deg, 0.5))

    cos_max = float(np.cos(theta_max / k + gs))
    cos_max = max(cos_max, 1e-10)
    sin_max = float(np.sin(theta_max))
    K = (R_max * sin_max / cos_max ** k) if sin_max > 1e-6 \
        else (R_max / np.cos(gs) ** k)

    theta_vals = np.linspace(1e-5, theta_max, N)

    cos_arg = np.cos(theta_vals / k + gs)
    cos_arg = np.maximum(cos_arg, 1e-10)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        r_vals = K / cos_arg ** k

    finite_mask = np.isfinite(r_vals) & (r_vals > 0)
    theta_vals  = theta_vals[finite_mask]
    r_vals      = r_vals[finite_mask]

    x_vals = r_vals * np.sin(theta_vals)
    z_vals = r_vals * np.cos(theta_vals)

    return x_vals, z_vals, theta_vals, r_vals, K, k, gs, theta_max_deg


def compute_virtual_surfaces(theta_vals, r_vals, k, gs, alpha):
    """
    Поверхности виртуального изображения (Stürzl et al., ур. 21, 23–24).
    """
    if abs(alpha - 1.0) < 1e-4:
        lam_t = np.full_like(r_vals, 1e6)
    else:
        lam_t = r_vals / (alpha - 1.0)

    denom = np.sin(alpha * theta_vals + 2 * gs) - np.sin(theta_vals)
    sign  = np.where(denom >= 0, 1.0, -1.0)
    denom = np.where(np.abs(denom) < 1e-8, sign * 1e-8, denom)
    lam_p = r_vals * np.sin(theta_vals) / denom

    ex = np.sin(theta_vals)
    ez = np.cos(theta_vals)
    xm = r_vals * ex
    zm = r_vals * ez

    vt_x = xm + lam_t * ex;  vt_z = zm + lam_t * ez
    vp_x = xm + lam_p * ex;  vp_z = zm + lam_p * ez

    return vt_x, vt_z, vp_x, vp_z


def blur_size(theta_vals, r_vals, k, gs, alpha, f_mm, D_mm):
    """
    Оценка Δξ (размытие) по ур. 27–29 (Stürzl et al.).
    """
    if abs(alpha - 1.0) < 1e-4:
        return np.zeros_like(r_vals)

    ztheta = r_vals * np.cos(theta_vals) * alpha / (alpha - 1.0)

    denom  = np.sin(alpha * theta_vals + 2 * gs) - np.sin(theta_vals)
    denom  = np.where(np.abs(denom) < 1e-8, 1e-8, denom)
    zphi   = r_vals * np.cos(theta_vals) * np.sin(alpha * theta_vals + 2 * gs) / denom

    safe_zp = np.where(np.abs(zphi) < 1e-8, 1e-8, zphi)
    safe_zt = np.where(np.abs(ztheta) < 1e-8, 1e-8, ztheta)

    ratio    = f_mm * (ztheta - zphi) / (safe_zp * safe_zt)
    delta_xi = np.abs(ratio) * D_mm
    delta_xi = np.where(np.isfinite(delta_xi), delta_xi, 0.0)
    return delta_xi

# ---------------------------------------------------------------------------
# Новое в v2.3: расчёт геометрии установки
# ---------------------------------------------------------------------------

def compute_mounting_geometry(x_vals, z_vals, fov_deg, sensor_mm, f_mm):
    """
    Расчёт параметров установки зеркала относительно камеры.
      • beta_max_deg — максимальный угол до края зеркала (от оптической оси).
      • H_min        — минимальная высота установки зеркала над камерой (мм),
                       чтобы зеркало целиком помещалось в FOV.
      • img_diameter — диаметр изображения зеркала на сенсоре при заданном f.
      • coverage_percent — заполнение сенсора (%).
      • f_recommended — фокусное расстояние для 100% заполнения сенсора.
    """
    x_vals = np.asarray(x_vals); z_vals = np.asarray(z_vals)
    mask = np.isfinite(x_vals) & np.isfinite(z_vals) & (z_vals > 1e-6)
    if not np.any(mask):
        return None
    x = x_vals[mask]; z = z_vals[mask]

    beta = np.arctan2(x, z)
    beta_max = float(np.max(beta))
    idx_max = int(np.argmax(beta))
    x_max_pt = float(x[idx_max])
    z_max_pt = float(z[idx_max])

    fov_rad = deg2rad(fov_deg)
    half_fov = fov_rad / 2.0

    if half_fov < 1e-6:
        H_min = float('inf')
    else:
        H_req = x_max_pt / np.tan(half_fov) - z_max_pt
        H_min = float(max(H_req, 0.0))

    if beta_max >= pi/2 - 1e-6:
        img_diameter = float('inf')
        f_rec = float('inf')
    else:
        img_diameter = 2.0 * f_mm * np.tan(beta_max)
        f_rec = (sensor_mm / 2.0) / np.tan(beta_max) if beta_max > 1e-6 else float('inf')

    coverage = (img_diameter / sensor_mm) * 100.0 if sensor_mm > 1e-6 else 0.0

    return {
        'beta_max_deg': beta_max * 180.0 / pi,
        'H_min': H_min,
        'img_diameter': img_diameter,
        'coverage_percent': coverage,
        'f_recommended': f_rec
    }

# ---------------------------------------------------------------------------
# Вспомогательная фильтрация для DXF / графика
# ---------------------------------------------------------------------------

def _filter_finite(xs, zs, xlim=300, zlim=500):
    """Убирает нефинитные и выбивающиеся значения."""
    m = (np.isfinite(xs) & np.isfinite(zs) &
         (np.abs(xs) < xlim) & (np.abs(zs) < zlim) & (zs > -20))
    return xs[m], zs[m]

# ---------------------------------------------------------------------------
# Начальные параметры
# ---------------------------------------------------------------------------

W0   = 35.0;  A0  = 120.0; B0  = 0.0;  RA0 = 0.0

ALPHA0 = 4.0; GS0 = 17.5;  RMAX0 = 23.0; THMAX0 = 26.0
F0  = 2.8;    D0  = 1.0
FOV0 = 54.0;  SENSOR0 = 2.88   # <-- новое в v2.3

state = compute_hyp_params(A0, B0, W0, RA0)
state.update({'w': W0, 'mode': 'Гиперболоид'})

N_RAYS = 9   # количество трассируемых лучей (нечётное — для симметрии)

# ---------------------------------------------------------------------------
# Фигура и оси
# ---------------------------------------------------------------------------

AXIS_COLOR = 'lightgoldenrodyellow'
EQ_COLOR   = 'lightcyan'
CAM_COLOR  = 'lightyellow'
RAY_COLOR_INC  = '#00aa00'   # падающий луч
RAY_COLOR_REF  = '#aa00aa'   # отражённый луч

fig = plt.figure(figsize=(14, 13))
ax  = fig.add_subplot(111)
fig.subplots_adjust(left=0.25, bottom=0.55)

ax.set_title("Zerkala v2.3 — Проектирование зеркал", fontsize=13, pad=10)
ax.set_xlim([-100, 100])
ax.set_ylim([-25, 175])
ax.set_xlabel("x (мм)")
ax.set_ylabel("z / y (мм)")
ax.grid(True, alpha=0.25, linestyle=':')

# ── объекты гиперболоида ──────────────────────────────────────────────────

t0 = arange(-W0, W0, 0.1)
p  = state

[camLine]  = ax.plot([-W0, 0],       [p['cr'][1], 0],         lw=2, ls='--', color='#1a6fe0')
[camLine2] = ax.plot([0,   W0],      [0, p['cr'][1]],         lw=2, ls='--', color='#1a6fe0')
[camLine3] = ax.plot([0,   p['rx']], [0, p['ry']],            lw=2, ls='--', color='#e07a1a')
[camLine4] = ax.plot([p['rx'], 101*p['rx']],
                     [p['ry'], 101*p['ry'] - 100*p['F2'][1]], lw=2, ls='--', color='#e07a1a')

[hyp_line] = ax.plot(t0, signal(p['a'], p['b'], t0, p['ei']), lw=2.5, color='#d32f2f',
                     label='Гиперболоид')
y_top_h    = (p['a'] * sqrt(p['b']**2 + W0**2)) / p['b'] + p['ei']
[hyp_top]  = ax.plot([-W0, W0], [y_top_h, y_top_h], lw=2, color='#d32f2f')

# Лучи для гиперболоида (новое в v2.3)
hyp_incident_rays = []
hyp_reflected_rays = []
for _ in range(N_RAYS):
    li, = ax.plot([], [], '-', color=RAY_COLOR_INC, lw=1.2, alpha=0.6)
    lr, = ax.plot([], [], '-', color=RAY_COLOR_REF, lw=1.2, alpha=0.6)
    hyp_incident_rays.append(li)
    hyp_reflected_rays.append(lr)

# ── объекты эквиуглового зеркала ─────────────────────────────────────────

_eq = compute_equiangular(ALPHA0, GS0, RMAX0, THMAX0)
eq_x0, eq_z0, eq_th0, eq_r0, _K, _k, _gs, _thused = _eq

[eq_right] = ax.plot(eq_x0,  eq_z0, lw=2.5, color='#2e7d32', visible=False, label='Эквиугловое')
[eq_left]  = ax.plot(-eq_x0, eq_z0, lw=2.5, color='#2e7d32', visible=False)

vt_x0, vt_z0, vp_x0, vp_z0 = compute_virtual_surfaces(eq_th0, eq_r0, _k, _gs, ALPHA0)

[vt_right] = ax.plot(vt_x0,  vt_z0, lw=1.2, ls='--', color='#7b1fa2', visible=False,
                     label='vθ (виртуальное θ)')
[vt_left]  = ax.plot(-vt_x0, vt_z0, lw=1.2, ls='--', color='#7b1fa2', visible=False)
[vp_right] = ax.plot(vp_x0,  vp_z0, lw=1.2, ls=':',  color='#00838f', visible=False,
                     label='vφ (виртуальное φ)')
[vp_left]  = ax.plot(-vp_x0, vp_z0, lw=1.2, ls=':',  color='#00838f', visible=False)

# Лучи для эквиуглового зеркала (новое в v2.3)
eq_incident_rays = []
eq_reflected_rays = []
for _ in range(N_RAYS):
    li, = ax.plot([], [], '-', color=RAY_COLOR_INC, lw=1.2, alpha=0.6, visible=False)
    lr, = ax.plot([], [], '-', color=RAY_COLOR_REF, lw=1.2, alpha=0.6, visible=False)
    eq_incident_rays.append(li)
    eq_reflected_rays.append(lr)

ax.legend(loc='upper right', fontsize=8, framealpha=0.8)

# Текстовый блок с параметрами установки (новое в v2.3)
mount_text = fig.text(
    0.02, 0.01, '',
    fontsize=9, family='monospace',
    verticalalignment='bottom',
    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6)
)

print(f"[Гиперболоид] Height: {p['cr'][1]:.4f}  Толщина: {p['cr'][1] - p['ry']:.4f}")

# ---------------------------------------------------------------------------
# Слайдеры
# ---------------------------------------------------------------------------

def _sl(pos, label, lo, hi, val, color=AXIS_COLOR):
    return Slider(fig.add_axes(pos, facecolor=color), label, lo, hi, valinit=val)

# Гиперболоид
a_sl  = _sl([0.25, 0.50, 0.65, 0.022], 'a · угол конуса (°)',  1,   179, A0)
b_sl  = _sl([0.25, 0.47, 0.65, 0.022], 'b · угол пучка (°)',   0,   359, B0)
w_sl  = _sl([0.25, 0.44, 0.65, 0.022], 'w · полуширина (мм)',  1,   100, W0)
ra_sl = _sl([0.25, 0.41, 0.65, 0.022], 'ra· угол луча (°)', -180,  180, RA0)

# Эквиугловое
alpha_sl = _sl([0.25, 0.36, 0.65, 0.022], 'α  · угл. увеличение',    1.01, 8.0,  ALPHA0, EQ_COLOR)
gs_sl    = _sl([0.25, 0.33, 0.65, 0.022], 'γS · угол вершины (°)',    0.0,  45.0, GS0,    EQ_COLOR)
rmax_sl  = _sl([0.25, 0.30, 0.65, 0.022], 'Rmax · макс. радиус (мм)', 2.0,  80.0, RMAX0,  EQ_COLOR)
thmax_sl = _sl([0.25, 0.27, 0.65, 0.022], 'θmax · макс. угол (°)',    2.0,  80.0, THMAX0, EQ_COLOR)

# Камера + геометрия установки (новое в v2.3)
f_sl      = _sl([0.25, 0.22, 0.65, 0.022], 'f  · фокус. расст. (мм)', 1.0, 30.0, F0, CAM_COLOR)
D_sl      = _sl([0.25, 0.19, 0.65, 0.022], 'D  · апертура (мм)',       0.1, 15.0, D0, CAM_COLOR)
fov_sl    = _sl([0.25, 0.16, 0.65, 0.022], 'FOV· угол обзора камеры(°)', 10, 180, FOV0, CAM_COLOR)
sensor_sl = _sl([0.25, 0.13, 0.65, 0.022], 'Sensor· размер сенсора(мм)', 1,  50,  SENSOR0, CAM_COLOR)

# ---------------------------------------------------------------------------
# Функции обновления графика
# ---------------------------------------------------------------------------

def update_hyperboloid():
    p = state;  w = p['w']
    t = arange(-w, w, 0.1)
    hyp_line.set_data(t, signal(p['a'], p['b'], t, p['ei']))
    y_top = (p['a'] * sqrt(p['b']**2 + w**2)) / p['b'] + p['ei']
    hyp_top.set_data([-w, w], [y_top, y_top])
    camLine.set_data([-w, 0],       [p['cr'][1], 0])
    camLine2.set_data([0, w],       [0, p['cr'][1]])
    camLine3.set_data([0, p['rx']], [0, p['ry']])
    camLine4.set_data([p['rx'], 101*p['rx']],
                      [p['ry'], 101*p['ry'] - 100*p['F2'][1]])

    # --- трассировка лучей гиперболоида (новое в v2.3) ---
    F1 = p['F1']; F2 = p['F2']
    t_rays = np.linspace(-w, w, N_RAYS)
    for i, t in enumerate(t_rays):
        y = signal(p['a'], p['b'], t, p['ei'])
        hyp_incident_rays[i].set_data([F1[0], t], [F1[1], y])
        dx = F2[0] - t
        dy = F2[1] - y
        norm = np.sqrt(dx**2 + dy**2)
        if norm > 1e-9:
            dx /= norm; dy /= norm
        L = max(p['cr'][1] * 1.5, 80.0)
        hyp_reflected_rays[i].set_data([t, t + L*dx], [y, y + L*dy])

    # --- расчёт геометрии установки (новое в v2.3) ---
    x_prof = np.abs(t)
    z_prof = signal(p['a'], p['b'], t, p['ei'])
    geom = compute_mounting_geometry(x_prof, z_prof, fov_sl.val, sensor_sl.val, f_sl.val)
    update_mount_text(geom, 'Гиперболоид')


def update_equiangular():
    alpha  = alpha_sl.val;  gs_d  = gs_sl.val
    rmax   = rmax_sl.val;   thmax = thmax_sl.val
    f_mm   = f_sl.val;      D_mm  = D_sl.val

    result = compute_equiangular(alpha, gs_d, rmax, thmax)
    xv, zv, thv, rv, K, k, gs, th_used = result

    finite_xy = np.isfinite(xv) & np.isfinite(zv)
    if not np.any(finite_xy):
        print("[Предупреждение] Нет валидных точек профиля — пропускаем обновление.")
        return

    xv_ok = xv[finite_xy];  zv_ok = zv[finite_xy]

    eq_right.set_data(xv_ok,  zv_ok)
    eq_left.set_data(-xv_ok,  zv_ok)

    vt_x, vt_z, vp_x, vp_z = compute_virtual_surfaces(thv, rv, k, gs, alpha)

    fx, fz = _filter_finite(vt_x, vt_z)
    vt_right.set_data(fx, fz);  vt_left.set_data(-fx, fz)

    fx, fz = _filter_finite(vp_x, vp_z)
    vp_right.set_data(fx, fz);  vp_left.set_data(-fx, fz)

    xmax_raw = float(np.nanmax(np.abs(xv_ok)))
    zmax_raw = float(np.nanmax(zv_ok))

    if not np.isfinite(xmax_raw) or xmax_raw < 1e-3:
        xmax_raw = 30.0
    if not np.isfinite(zmax_raw) or zmax_raw < 1e-3:
        zmax_raw = 30.0

    xmax = max(xmax_raw * 1.35, 15.0)
    zmax = max(zmax_raw * 1.25, 15.0)

    ax.set_xlim([-xmax, xmax])
    ax.set_ylim([-8, zmax])

    # --- трассировка лучей эквиуглового зеркала (новое в v2.3) ---
    if len(thv) > 0:
        for i in range(N_RAYS):
            idx = int(len(thv) * i / max(N_RAYS - 1, 1))
            idx = min(idx, len(thv) - 1)
            x_pt = float(xv[idx]); z_pt = float(zv[idx])
            th_pt = float(thv[idx])

            eq_incident_rays[i].set_data([0, x_pt], [0, z_pt])

            # Формула (17) из Stürzl et al.: theta_o = pi - 2*gamma_S - alpha*theta
            theta_o = pi - 2.0 * gs - alpha * th_pt
            dx = np.sin(theta_o)
            dz = np.cos(theta_o)
            L = max(zmax * 1.2, 50.0)
            eq_reflected_rays[i].set_data([x_pt, x_pt + L*dx], [z_pt, z_pt + L*dz])

    dxi      = blur_size(thv, rv, k, gs, alpha, f_mm, D_mm)
    mean_bl  = float(np.nanmean(dxi)) * 1e3   # µm
    print(f"[Эквиугл.] α={alpha:.2f}  γS={gs_d:.1f}°  "
          f"Rmax={rmax:.1f}  θmax={th_used:.1f}° (лимит)  "
          f"K={K:.4f}  k={k:.4f}  <Δξ>≈{mean_bl:.1f} µm")

    # --- расчёт геометрии установки (новое в v2.3) ---
    geom = compute_mounting_geometry(xv_ok, zv_ok, fov_sl.val, sensor_sl.val, f_sl.val)
    update_mount_text(geom, 'Эквиугловое')


def update_mount_text(geom, mode_name):
    if geom is None:
        mount_text.set_text('Нет данных для расчёта геометрии установки')
        return
    txt = (
        f"Режим: {mode_name}\n"
        f"Макс. угол зеркала: {geom['beta_max_deg']:.2f}°\n"
        f"Мин. высота установки H_min: {geom['H_min']:.2f} мм\n"
        f"Диам. изображения на сенсоре: {geom['img_diameter']:.2f} мм\n"
        f"Заполнение сенсора: {geom['coverage_percent']:.1f}%\n"
        f"Реком. фокусное расстояние: {geom['f_recommended']:.2f} мм"
    )
    mount_text.set_text(txt)


def _set_visibility(hyp_on):
    for art in [hyp_line, hyp_top, camLine, camLine2, camLine3, camLine4]:
        art.set_visible(hyp_on)
    for art in [eq_right, eq_left, vt_right, vt_left, vp_right, vp_left]:
        art.set_visible(not hyp_on)
    # --- видимость лучей (новое в v2.3) ---
    for art in hyp_incident_rays + hyp_reflected_rays:
        art.set_visible(hyp_on)
    for art in eq_incident_rays + eq_reflected_rays:
        art.set_visible(not hyp_on)


def set_mode(mode):
    state['mode'] = mode
    if mode == 'Гиперболоид':
        _set_visibility(True)
        ax.set_xlim([-100, 100]);  ax.set_ylim([-25, 175])
        update_hyperboloid()
    else:
        _set_visibility(False)
        update_equiangular()
    fig.canvas.draw_idle()

# ---------------------------------------------------------------------------
# Колбэки слайдеров
# ---------------------------------------------------------------------------

def on_hyp(_v):
    if state['mode'] != 'Гиперболоид':
        return
    w   = w_sl.val
    new = compute_hyp_params(a_sl.val, b_sl.val, w, ra_sl.val)
    state.update(new);  state['w'] = w
    print(f"[Гиперболоид] Height: {state['cr'][1]:.4f}  "
          f"Толщина: {state['cr'][1] - state['ry']:.4f}")
    update_hyperboloid();  fig.canvas.draw_idle()


def on_eq(_v):
    if state['mode'] != 'Эквиугловое':
        return
    update_equiangular();  fig.canvas.draw_idle()


def on_cam(_v):
    # FOV и размер сенсора тоже влияют на текст/лучи
    if state['mode'] == 'Эквиугловое':
        update_equiangular();  fig.canvas.draw_idle()
    else:
        update_hyperboloid();  fig.canvas.draw_idle()


for sl in (a_sl, b_sl, w_sl, ra_sl):
    sl.on_changed(on_hyp)
for sl in (alpha_sl, gs_sl, rmax_sl, thmax_sl):
    sl.on_changed(on_eq)
for sl in (f_sl, D_sl, fov_sl, sensor_sl):
    sl.on_changed(on_cam)

# ---------------------------------------------------------------------------
# Радиокнопки
# ---------------------------------------------------------------------------

f_radios = RadioButtons(
    fig.add_axes([0.025, 0.60, 0.15, 0.10], facecolor=AXIS_COLOR),
    ('Гиперболоид', 'Эквиугловое'), active=0)
f_radios.on_clicked(set_mode)

# ---------------------------------------------------------------------------
# Кнопка «Сохранить STL»
# ---------------------------------------------------------------------------

btn_stl = Button(fig.add_axes([0.025, 0.79, 0.15, 0.055], facecolor=AXIS_COLOR),
                 'Сохранить\nSTL')


def save_stl(_ev):
    N_TH = 180
    th_c = np.linspace(0, 2 * pi, N_TH, endpoint=False)

    if state['mode'] == 'Гиперболоид':
        w_p  = state['w'];  N_Z = 300
        zz   = np.linspace(-w_p, w_p, N_Z)
        R    = signal(state['a'], state['b'], zz, state['ei'])
        fname = 'zerkalo_hyperboloid.stl'
    else:
        xv, zv, *_ = compute_equiangular(
            alpha_sl.val, gs_sl.val, rmax_sl.val, thmax_sl.val, N=300)
        R = xv;  zz = zv;  N_Z = len(R)
        fname = 'zerkalo_equiangular.stl'

    R2D = R[:, np.newaxis]
    TH  = th_c[np.newaxis, :]
    X   = R2D * np.cos(TH);  Y = R2D * np.sin(TH)
    Z   = np.tile(zz[:, np.newaxis], (1, N_TH))

    tris = []
    for i in range(N_Z - 1):
        for j in range(N_TH):
            j1 = (j + 1) % N_TH
            tris.append([[X[i,j],Y[i,j],Z[i,j]],
                         [X[i+1,j],Y[i+1,j],Z[i+1,j]],
                         [X[i,j1],Y[i,j1],Z[i,j1]]])
            tris.append([[X[i+1,j],Y[i+1,j],Z[i+1,j]],
                         [X[i+1,j1],Y[i+1,j1],Z[i+1,j1]],
                         [X[i,j1],Y[i,j1],Z[i,j1]]])

    tris = np.array(tris)
    m    = mesh.Mesh(np.zeros(len(tris), dtype=mesh.Mesh.dtype))
    m.vectors = tris
    m.save(fname)
    print(f"STL сохранён: {fname}  ({len(tris)} треугольников)")


btn_stl.on_clicked(save_stl)

# ---------------------------------------------------------------------------
# Кнопка «Сохранить DXF»
# ---------------------------------------------------------------------------

btn_dxf = Button(fig.add_axes([0.025, 0.72, 0.15, 0.055], facecolor='lightcyan'),
                 'Сохранить\nDXF')


def _new_doc():
    doc = ezdxf.new(dxfversion='R2010')
    doc.header['$INSUNITS'] = 4
    if 'DASHED' not in doc.linetypes:
        doc.linetypes.new('DASHED', dxfattribs={
            'description': 'Dashed', 'pattern': [6.0, 4.0, -2.0]})
    return doc, doc.modelspace()


def save_dxf(_ev):
    if state['mode'] == 'Гиперболоид':
        _save_dxf_hyp()
    else:
        _save_dxf_eq()


def _save_dxf_hyp():
    doc, msp = _new_doc()
    doc.layers.new('HYPERBOLA',   dxfattribs={'color': 1})
    doc.layers.new('TOP_LINE',    dxfattribs={'color': 1})
    doc.layers.new('CAM_LINES',   dxfattribs={'color': 5})
    doc.layers.new('RAY_LINES',   dxfattribs={'color': 30})
    doc.layers.new('ANNOTATIONS', dxfattribs={'color': 7})

    a_p, b_p, ei_p = state['a'], state['b'], state['ei']
    w_p            = state['w']
    cr_p, F2_p     = state['cr'], state['F2']
    rx_p, ry_p     = state['rx'], state['ry']

    t_pts = np.arange(-w_p, w_p + 0.1, 0.1)
    y_pts = signal(a_p, b_p, t_pts, ei_p)
    msp.add_lwpolyline(list(zip(t_pts.tolist(), y_pts.tolist())),
                       dxfattribs={'layer': 'HYPERBOLA'})

    y_top = (a_p * sqrt(b_p**2 + w_p**2)) / b_p + ei_p
    msp.add_line((-w_p, y_top), (w_p, y_top),  dxfattribs={'layer': 'TOP_LINE'})
    msp.add_line((-w_p, cr_p[1]), (0, 0),       dxfattribs={'layer': 'CAM_LINES', 'linetype': 'DASHED'})
    msp.add_line((0, 0), (w_p, cr_p[1]),         dxfattribs={'layer': 'CAM_LINES', 'linetype': 'DASHED'})
    msp.add_line((0, 0), (rx_p, ry_p),           dxfattribs={'layer': 'RAY_LINES', 'linetype': 'DASHED'})
    sc = 50.0
    msp.add_line((rx_p, ry_p),
                 (rx_p + sc*rx_p, ry_p + sc*(ry_p - F2_p[1])),
                 dxfattribs={'layer': 'RAY_LINES', 'linetype': 'DASHED'})

    h = cr_p[1];  thick = cr_p[1] - ry_p
    msp.add_text(f"H={h:.2f}  T={thick:.2f}",
                 dxfattribs={'layer': 'ANNOTATIONS', 'height': 2,
                             'insert': (w_p + 3, y_top)})
    msp.add_mtext(f"a={a_p:.3f}  b={b_p:.3f}  ei={ei_p:.3f}\\Pw={w_p:.1f}",
                  dxfattribs={'layer': 'ANNOTATIONS', 'char_height': 2,
                              'insert': (-w_p, -10)})
    fname = 'zerkalo_hyperboloid.dxf'
    doc.saveas(fname)
    print(f"DXF (Гиперболоид) сохранён: {fname}  (точек профиля: {len(t_pts)})")


def _save_dxf_eq():
    doc, msp = _new_doc()
    doc.layers.new('MIRROR_PROFILE', dxfattribs={'color': 3})
    doc.layers.new('VIRTUAL_THETA',  dxfattribs={'color': 6})
    doc.layers.new('VIRTUAL_PHI',    dxfattribs={'color': 4})
    doc.layers.new('AXIS',           dxfattribs={'color': 8})
    doc.layers.new('ANNOTATIONS',    dxfattribs={'color': 7})

    alpha = alpha_sl.val;  gs_d  = gs_sl.val
    rmax  = rmax_sl.val;   thmax = thmax_sl.val
    f_mm  = f_sl.val;      D_mm  = D_sl.val

    xv, zv, thv, rv, K, k, gs, th_used = compute_equiangular(alpha, gs_d, rmax, thmax)
    vt_x, vt_z, vp_x, vp_z = compute_virtual_surfaces(thv, rv, k, gs, alpha)

    for sx in (1, -1):
        msp.add_lwpolyline(list(zip((sx*xv).tolist(), zv.tolist())),
                           dxfattribs={'layer': 'MIRROR_PROFILE'})

    for sx in (1, -1):
        fx, fz = _filter_finite(vt_x, vt_z)
        if len(fx) > 1:
            msp.add_lwpolyline(list(zip((sx*fx).tolist(), fz.tolist())),
                               dxfattribs={'layer': 'VIRTUAL_THETA', 'linetype': 'DASHED'})
        fx, fz = _filter_finite(vp_x, vp_z)
        if len(fx) > 1:
            msp.add_lwpolyline(list(zip((sx*fx).tolist(), fz.tolist())),
                               dxfattribs={'layer': 'VIRTUAL_PHI', 'linetype': 'DASHED'})
    z_bot = float(np.min(zv));  z_top = float(np.max(zv))
    msp.add_line((0, z_bot - 5), (0, z_top + 5),
                 dxfattribs={'layer': 'AXIS', 'linetype': 'DASHED'})

    dxi     = blur_size(thv, rv, k, gs, alpha, f_mm, D_mm)
    mean_bl = float(np.nanmean(dxi)) * 1e3
    msp.add_mtext(
        f"Эквиугловое зеркало (Sturzl et al., 2004)\\P"
        f"alpha={alpha:.2f}  gS={gs_d:.1f}deg  Rmax={rmax:.1f}mm  "
        f"thetamax={th_used:.1f}deg\\P"
        f"f={f_mm:.1f}mm  D={D_mm:.1f}mm\\P"
        f"K={K:.4f}  k={k:.4f}\\P"
        f"<Delta_xi> approx {mean_bl:.1f} um",
        dxfattribs={'layer': 'ANNOTATIONS', 'char_height': 1.5,
                    'insert': (float(np.max(xv)) + 3, z_top)})

    fname = 'zerkalo_equiangular.dxf'
    doc.saveas(fname)
    print(f"DXF (Эквиугловое) сохранён: {fname}  "
          f"(точек: {len(xv)}, θmax_eff={th_used:.1f}°, <Δξ>≈{mean_bl:.1f} µm)")


btn_dxf.on_clicked(save_dxf)

# ---------------------------------------------------------------------------
# Создание файла инструкции (новое в v2.3)
# ---------------------------------------------------------------------------

INSTRUCTION_TEXT = """\
================================================================================
  ИНСТРУКЦИЯ ПО ИСПОЛЬЗОВАНИЮ ZERKALA v2.3
================================================================================

1. НАЗНАЧЕНИЕ
   Программа предназначена для проектирования осесимметричных катадиоптрических
   зеркал: гиперболоидных и эквиугловых (equi-angular). Позволяет визуально
   оценить профиль зеркала, виртуальные поверхности, качество фокусировки,
   а также рассчитать параметры установки зеркала относительно камеры.

2. РЕЖИМЫ РАБОТЫ
   Переключение режимов осуществляется радиокнопками в левой части окна.

   2.1 Гиперболоид
       Зеркало строится на основе двух фокусов (F1 — камера, F2 — виртуальный).
       Параметры:
         • a — угол конуса (°)
         • b — угол пучка (°)
         • w — полуширина зеркала (мм)
         • ra — угол луча (°), задаёт точку на поверхности для трассировки

   2.2 Эквиугловое зеркало (Stürzl et al., ECCV 2004)
       Обеспечивает постоянное угловое увеличение α.
       Параметры:
         • α  — угловое увеличение
         • γS — угол вершины (°)
         • Rmax — максимальный радиус зеркала (мм)
         • θmax — максимальный угол профиля (°)

3. ПАРАМЕТРЫ КАМЕРЫ И УСТАНОВКИ (новое в v2.3)
   Расположены в нижней группе слайдеров:
         • f  — фокусное расстояние объектива (мм)
         • D  — диаметр входного зрачка (апертура) (мм)
         • FOV — угол обзора камеры (°)
         • Sensor — размер сенсора (мм, диагональ или ширина)

   На основании этих параметров программа автоматически рассчитывает:
         • Минимальную высоту установки H_min — насколько нужно поднять зеркало
           над камерой (вдоль оптической оси), чтобы оно целиком помещалось
           в поле зрения.
         • Диаметр изображения зеркала на сенсоре при заданном f.
         • Процент заполнения сенсора.
         • Рекомендуемое фокусное расстояние для 100%-го заполнения сенсора.

4. ТРАССИРОВКА ЛУЧЕЙ (новое в v2.3)
   На графике отображаются 9 лучей:
         • Зелёные — падающие из точки фокуса (камеры) на профиль зеркала.
         • Пурпурные — отражённые лучи.
   Для гиперболоида отражённые лучи направлены ко второму фокусу.
   Для эквиуглового зеркала направление отражённого луча рассчитывается по
   формуле Стёрцла: θo = π − 2·γS − α·θ.

5. ВИРТУАЛЬНЫЕ ПОВЕРХНОСТИ (только эквиугловое зеркало)
   Пунктирные кривые показывают виртуальные изображения:
         • vθ (фиолетовый пунктир) — фокусировка на вертикальные структуры.
         • vφ (бирюзовый пунктир) — фокусировка на горизонтальные структуры.
   Разнос этих поверхностей определяет геометрическое размытие.

6. СОХРАНЕНИЕ
   • «Сохранить STL» — экспорт 3D-модели зеркала в формат STL для печати
     или ЧПУ-обработки.
   • «Сохранить DXF» — экспорт 2D-профиля и вспомогательных линий в DXF
     для импорта в AutoCAD / Компас / SolidWorks.

7. РЕКОМЕНДАЦИИ ПО ЭКСПЛУАТАЦИИ
   • Если H_min > 0, значит при текущем FOV зеркало не помещается в кадр
     вплотную к камере — его нужно отодвинуть вверх минимум на H_min мм.
   • Если заполнение сенсора > 100%, изображение зеркала не влезает на матрицу
     — увеличьте f или уменьшите зеркало.
   • Для стереосистем (два зеркала) используйте одинаковое α и подбирайте
     γS так, чтобы виртуальные поверхности vθ обоих зеркал были близки
     друг к другу — это минимизирует размытие стереопары.

8. ТРЕБОВАНИЯ
   Python 3, numpy, matplotlib, numpy-stl, ezdxf.

================================================================================
  Программа от команды НЕЧТО Zerkala Project. Версия 2.3.
================================================================================
"""

instr_path = 'INSTRUKCIYA_ZERKALA.txt'
if not os.path.exists(instr_path):
    try:
        with open(instr_path, 'w', encoding='utf-8') as f:
            f.write(INSTRUCTION_TEXT)
        print(f"Инструкция создана: {instr_path}")
    except Exception as e:
        print(f"Не удалось создать инструкцию: {e}")
else:
    print(f"Инструкция уже существует: {instr_path}")

# ---------------------------------------------------------------------------
# Первичная инициализация
# ---------------------------------------------------------------------------
update_hyperboloid()

# ---------------------------------------------------------------------------
plt.show()
