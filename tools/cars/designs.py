"""The three original car designs (parameters + per-car features)."""
import numpy as np
import sdf
from sdf import smin, smax, sstep, Profile, gauss, window
import meshops
import bl
import carlib
from carlib import CarBase, MID, piece, sdf_part


def PF(fn):
    """Adapt fn(X, Y, Z) to a field over an (N,3) point array."""
    return lambda P: fn(P[:, 0], P[:, 1], P[:, 2])


def sym_poly(half):
    """Mirror a half polygon [(x>=0, v), ...] (listed from the centreline outwards and back) about x=0."""
    return list(half) + [(-x, v) for (x, v) in reversed(half) if x > 1e-9]


class Aurion(CarBase):
    """AURION S8 - mid-engine supercar: low wedge nose, cab-forward teardrop canopy,
    side intakes ahead of muscular rear haunches."""
    id = 'aurion'
    name = 'AURION S8'
    paint_color = (0.55, 0.03, 0.03)

    def setup(self):
        self.axle_f, self.axle_r = 1.35, -1.35
        self.track_f, self.track_r = 1.68, 1.66
        self.rf, self.rr = 0.335, 0.350
        self.wf, self.wr = 0.25, 0.31
        self.half_w = 0.99
        self.z_nose, self.z_tail = 2.38, -2.24
        self.height = 1.19
        self.rim_style = 'twin5'
        self.k_shoulder, self.k_nose, self.k_tail, self.k_sill = 0.05, 0.055, 0.045, 0.035
        self.lean_up = 0.07
        self.nose_plan, self.nose_pow, self.nose_vert, self.nose_tip_y = 0.46, 2.1, 0.8, 0.30
        self.tail_plan, self.tail_pow, self.tail_tuck, self.tail_tuck_y = 0.26, 3.0, 0.07, (0.50, 0.26)
        self.p_hood = Profile([(2.38, 0.36), (2.20, 0.415), (1.85, 0.52), (1.40, 0.625), (1.05, 0.705),
                               (0.85, 0.755), (0.30, 0.82), (-0.50, 0.86), (-1.20, 0.895), (-1.80, 0.91),
                               (-2.05, 0.915), (-2.24, 0.915)])
        self.p_fender = Profile([(2.38, 0.36), (2.22, 0.47), (1.95, 0.655), (1.62, 0.755), (1.35, 0.782),
                                 (1.00, 0.782), (0.50, 0.775), (0.00, 0.785), (-0.45, 0.82), (-0.90, 0.89),
                                 (-1.35, 0.935), (-1.80, 0.94), (-2.05, 0.93), (-2.24, 0.925)])
        self.p_width = Profile([(2.38, 0.70), (2.10, 0.92), (1.75, 0.972), (1.35, 0.985), (1.00, 0.968),
                                (0.50, 0.935), (0.00, 0.92), (-0.45, 0.94), (-0.90, 0.98), (-1.35, 0.995),
                                (-1.75, 0.985), (-2.05, 0.955), (-2.24, 0.93)])
        self.p_ywide = Profile([(2.38, 0.40), (1.35, 0.50), (0.0, 0.50), (-1.35, 0.57), (-2.24, 0.62)])
        self.p_floor = Profile([(2.38, 0.135), (2.0, 0.122), (1.2, 0.115), (-1.0, 0.115), (-1.7, 0.125),
                                (-2.24, 0.20)])
        self.p_roof = Profile([(2.40, -0.30), (1.80, 0.15), (1.35, 0.52), (1.00, 0.74), (0.70, 0.93),
                               (0.40, 1.08), (0.15, 1.165), (-0.10, 1.205), (-0.40, 1.205), (-0.75, 1.155),
                               (-1.20, 1.05), (-1.65, 0.955), (-2.05, 0.895), (-2.30, 0.86)])
        self.p_crown = Profile([(1.4, 0.36), (0.4, 0.33), (-0.1, 0.22), (-1.0, 0.22), (-2.3, 0.30)])
        self.p_cwidth = Profile([(1.40, 0.52), (1.00, 0.68), (0.60, 0.765), (0.20, 0.80), (-0.30, 0.80),
                                 (-0.80, 0.745), (-1.30, 0.65), (-1.80, 0.55), (-2.30, 0.46)])
        self.cabin = dict(z_front=0.80, z_rear=-0.62, y_floor=0.17, seat_x=0.37, seat_z=-0.12)
        self.canopy_drop = 0.07

    # -------------------------------------------------------------- sculpting
    def side_detail(self, Y, Z):
        y_hi = 0.49 + 0.16 * sstep(0.90, -0.75, Z)
        win = sstep(0.20, 0.40, Y) * (1.0 - sstep(y_hi - 0.065, y_hi + 0.065, Y))
        ch = 0.10 * sstep(0.85, -0.45, Z) * (1.0 - sstep(-0.69, -0.79, Z)) * win
        return -ch

    HEAD_POLY = [(0.36, 2.290), (0.60, 2.212), (0.80, 2.078), (0.915, 1.865), (0.905, 1.775), (0.80, 1.865),
                 (0.70, 2.00), (0.56, 2.115), (0.37, 2.225)]
    HEAD_DRL = [(0.385, 2.238), (0.565, 2.135), (0.705, 2.022), (0.805, 1.895), (0.89, 1.80)]
    HEAD_PROJ = [(0.735, 2.078), (0.815, 1.985)]
    INTAKE_POLY = [(0.30, 0.17), (0.82, 0.19), (0.855, 0.30), (0.76, 0.355), (0.34, 0.30)]
    HOOD_VENT = [(0.17, 1.47), (0.45, 1.41), (0.45, 1.315), (0.17, 1.365)]
    DECK_VENT = [(0.62, -1.52), (0.84, -1.58), (0.83, -1.92), (0.66, -1.88)]
    HOOD_SEAM = sym_poly([(0.0, 2.20), (0.36, 2.13), (0.60, 1.74), (0.655, 1.20), (0.52, 1.04), (0.0, 1.06)])
    DOOR_SEAM = [(0.83, 0.235), (0.80, 0.60), (0.70, 0.95), (-0.50, 0.95), (-0.56, 0.60), (-0.52, 0.235)]

    def headlamp(self, X, Y, Z):
        d = sdf.sd_polygon(np.abs(X), Z, self.HEAD_POLY) - 0.010
        return np.maximum(d, 0.41 - Y)

    TAIL_POLY = sym_poly([(0.0, 0.835), (0.86, 0.835), (0.90, 0.80), (0.86, 0.765), (0.74, 0.74), (0.68, 0.50), (0.0, 0.50)])

    def tail_panel(self, X, Y, Z):
        """Full-width dark rear panel: light bar on top, mesh + exhausts below."""
        d = sdf.sd_polygon(X, Y, self.TAIL_POLY) - 0.022
        return np.maximum(d, Z + 1.8)

    def intake_outer(self, X, Y, Z):
        aX = np.abs(X)
        return np.maximum(sdf.sd_polygon(aX, Y, self.INTAKE_POLY) - 0.025, (self.front_face(aX, Y) - 0.22) - Z)

    def intake_centre(self, X, Y, Z):
        aX = np.abs(X)
        return np.maximum(sdf.sd_rrect(aX, Y, 0.0, 0.205, 0.24, 0.045, 0.03), (self.front_face(aX, Y) - 0.25) - Z)

    def duct(self, X, Y, Z):
        return sdf.sd_box(np.abs(X), Y, Z, (0.808, 0.49, -0.805), (0.068, 0.13, 0.105), 0.035)

    def tunnel(self, X, Y, Z):
        ramp = 0.115 + 0.285 * np.clip((-1.60 - Z) / 0.66, 0.0, 1.0)
        return smax(smax(np.abs(X) - 0.70, Y - ramp, 0.03), Z + 1.60, 0.03)

    def hood_vent(self, X, Y, Z):
        return np.maximum(sdf.sd_polygon(np.abs(X), Z, self.HOOD_VENT) - 0.012, 0.5 - Y)

    def deck_vent(self, X, Y, Z):
        return np.maximum(sdf.sd_polygon(np.abs(X), Z, self.DECK_VENT) - 0.015, 0.6 - Y)

    def cut_list(self):
        return [
            dict(fn=self.duct, k=0.015, mat='trim_black'),
            dict(fn=self.tunnel, k=0.02, mat='carbon'),
            dict(fn=self.intake_outer, k=0.02, mat='trim_black'),
            dict(fn=self.intake_centre, k=0.02, mat='trim_black'),
        ]

    # -------------------------------------------------------------- materials
    PW = 0.030

    def glass_top(self, P):
        hull = self.hull(P[:, 0], P[:, 1], P[:, 2])
        ts = self.canopy_ts(P[:, 0], P[:, 1], P[:, 2])
        return np.maximum(np.maximum(self.PW - ts, 0.055 - hull), -1.22 - P[:, 2])

    def glass_side(self, P):
        hull = self.hull(P[:, 0], P[:, 1], P[:, 2])
        ts = self.canopy_ts(P[:, 0], P[:, 1], P[:, 2])
        return np.maximum(np.maximum(ts + self.PW, 0.045 - hull), (-0.58 + 0.55 * (P[:, 1] - 0.85)) - P[:, 2])

    def glass_any(self, P):
        return np.minimum(self.glass_top(P), self.glass_side(P))

    def paint(self, skin, lod):
        self.paint_base(skin, lod)
        fine = lod == 0
        hull = PF(self.hull)
        ts = PF(self.canopy_ts)
        # carbon lower body: sills between the arches, chin and rear valance
        skin.region(lambda P: np.maximum(P[:, 1] - 0.205, np.abs(P[:, 2] - 0.01) - 0.955), 'carbon', only=['paint'])
        skin.region(lambda P: np.maximum(P[:, 1] - 0.168, 1.75 - P[:, 2]), 'carbon', only=['paint'])
        skin.region(lambda P: np.maximum(P[:, 1] - 0.43, P[:, 2] + 1.78), 'carbon', only=['paint'])
        # greenhouse (everything standing proud of the hull)
        gh = self.greenhouse
        skin.region(gh, 'trim_black', only=['paint'], refine=0.03 if fine else None)
        # canopy tail + sail panel back to body colour
        skin.region(lambda P: np.maximum(gh(P), P[:, 2] + 1.30), 'paint', only=['trim_black'])
        sail = lambda P: np.maximum(np.maximum(gh(P), ts(P) + self.PW),
                                    P[:, 2] - (-0.61 + 0.55 * (P[:, 1] - 0.85)))
        skin.region(sail, 'paint', only=['trim_black'], refine=0.03 if fine else None)
        skin.region(self.glass_top, 'glass', only=['trim_black'], refine=0.03 if fine else None)
        skin.region(self.glass_side, 'glass', only=['trim_black'], refine=0.03 if fine else None)
        if fine:
            for zc in (0.30, -0.62):
                skin.band(lambda P, zc=zc: P[:, 2] - zc, 0.016, 'trim_black', only=['glass'],
                          extent=lambda P: 0.02 - ts(P))
        # flush dark panels: lamp housings, vents, rear grille
        self.inlay(skin, self.headlamp, 0.016, fine)
        self.inlay(skin, self.tail_panel, 0.022, fine, ramp=0.014)
        if fine:
            self.inlay(skin, self.hood_vent, 0.016, fine)
            self.inlay(skin, self.deck_vent, 0.016, fine)
        # lamps

    # -------------------------------------------------------------- decals (seams, LED strips)
    def decals(self, dec, lod):
        fine = lod == 0
        eps_l = 0.0015 if fine else 0.004
        step = 0.02 if fine else 0.06
        # headlamp: DRL blade + two projector dots per side
        for sgn in (1, -1):
            pts = carlib.polyline_points([(sgn * x, z) for x, z in self.HEAD_DRL], step)
            dec.ribbon(dec.project_2d(pts, 'top'), 0.0085, 'light_head', eps=eps_l, only=['trim_black'])
            for cx, cz in self.HEAD_PROJ:
                h = dec.project_2d([(sgn * cx, cz)], 'top')[0]
                if h is not None:
                    dec.disc(h[0], 0.024, 'light_head', eps=eps_l, segs=14 if fine else 8)
        # tail: full-width light strip + brake blocks + high-mount brake strip
        xs = carlib.polyline_points([(-0.845, 0.812), (0.845, 0.812)], step)
        dec.ribbon(dec.project_2d(xs, 'rear'), 0.010, 'light_tail', eps=eps_l, only=['trim_black'])
        for sgn in (1, -1):
            xs = carlib.polyline_points([(sgn * 0.365, 0.786), (sgn * 0.835, 0.786)], step)
            dec.ribbon(dec.project_2d(xs, 'rear'), 0.008, 'light_brake', eps=eps_l, only=['trim_black'])
        xs = carlib.polyline_points([(-0.17, -1.262), (0.17, -1.262)], step)
        dec.ribbon(dec.project_2d(xs, 'top'), 0.008, 'light_brake', eps=eps_l)
        if not fine:
            return
        # panel seams
        hood = carlib.outline_points(self.HOOD_SEAM, round_r=0.03, step=0.025)
        dec.ribbon(dec.project_2d(hood, 'top'), 0.003, 'trim_black', eps=0.001, only=['paint'], closed=True)
        door = carlib.outline_points(self.DOOR_SEAM, round_r=0.04, step=0.025)
        for sgn in (1, -1):
            dec.ribbon(dec.project_2d(door, 'side', sgn), 0.003, 'trim_black', eps=0.001, only=['paint'], closed=True)

    # -------------------------------------------------------------- parts
    def parts(self, lod):
        out = []
        p, fn = carlib.part_splitter(self, 0.100, 0.140, ext=0.045, z_back=1.80, lod=lod)
        out.append(p); self.part_fields.append(fn)
        p, fn = carlib.part_skirt(self, -0.93, 0.95, 0.088, 0.122, lod=lod)
        out.append(p); self.part_fields.append(fn)
        ramp = lambda Z: 0.115 + 0.285 * np.clip((-1.60 - Z) / 0.66, 0.0, 1.0)
        p, fn = carlib.part_fins(self, [0.0, 0.233, 0.466], -1.66, -2.215, 0.105, 0.17, top_fn=ramp, lod=lod)
        out.append(p); self.part_fields.append(fn)
        p, fn = carlib.part_wing(zc=-2.08, yc=0.975, chord=0.25, span_half=0.72, thick=0.024, angle_deg=7.0,
                                 pylons=(0.40,), deck_y=0.93, endplate=0.0, lod=lod)
        out.append(p); self.part_fields.append(fn)
        ps, fn = carlib.part_mirrors(np.array([0.985, 0.868, 0.60]), (0.088, 0.052, 0.074),
                                     np.array([0.885, 0.795, 0.625]), np.array([0.935, 0.848, 0.61]), lod=lod)
        out += ps; self.part_fields.append(fn)
        out += carlib.part_exhaust(0.15, 0.585, -2.235, length=0.16, r_out=0.058, r_in=0.049, lod=lod)
        return out

    def interior(self, lod):
        c = self.cabin
        return carlib.build_interior(self, c['z_front'], c['z_rear'], c['y_floor'], c['seat_x'], c['seat_z'],
                                     self.glass_any, lod=lod)

    def points(self):
        c = self.cabin
        return {
            'headlightL': (0.74, 0.60, 2.06), 'headlightR': (-0.74, 0.60, 2.06),
            'taillightL': (0.62, 0.81, -2.20), 'taillightR': (-0.62, 0.81, -2.20),
            'exhaustL': (0.15, 0.585, -2.25), 'exhaustR': (-0.15, 0.585, -2.25),
            'camHood': (0.0, 0.92, 1.02), 'camCockpit': (c['seat_x'], 0.99, c['seat_z'] - 0.06),
            'camBumper': (0.0, 0.42, 2.41), 'wingCenter': (0.0, 0.975, -2.08),
        }


class Valken(CarBase):
    """VALKEN GT - front-engine grand tourer / muscle coupe: long hood with a power bulge,
    cab-rearward fastback, wide hips, ducktail."""
    id = 'valken'
    name = 'VALKEN GT'
    paint_color = (0.02, 0.07, 0.30)

    def setup(self):
        self.axle_f, self.axle_r = 1.375, -1.375
        self.track_f, self.track_r = 1.64, 1.65
        self.rf, self.rr = 0.345, 0.355
        self.wf, self.wr = 0.265, 0.305
        self.gap_f, self.gap_r = 0.055, 0.055
        self.half_w = 0.975
        self.z_nose, self.z_tail = 2.30, -2.48
        self.height = 1.31
        self.rim_style = 'split5'
        self.concave_f, self.concave_r = 0.04, 0.06
        self.k_shoulder, self.k_nose, self.k_tail, self.k_sill = 0.055, 0.06, 0.05, 0.04
        self.lean_up = 0.06
        self.hood_crown = 0.02
        self.fender_x0 = Profile([(2.3, 0.20), (0.6, 0.22), (0.0, 0.40), (-2.5, 0.45)])
        self.fender_x1 = Profile([(2.3, 0.42), (0.6, 0.44), (0.0, 0.70), (-2.5, 0.75)])
        self.nose_plan, self.nose_pow, self.nose_vert, self.nose_tip_y = 0.30, 2.6, 0.45, 0.50
        self.tail_plan, self.tail_pow, self.tail_tuck, self.tail_tuck_y = 0.24, 3.0, 0.09, (0.62, 0.36)
        self.tumble, self.belt_ref, self.canopy_drop = 0.40, 0.90, 0.06
        self.p_hood = Profile([(2.30, 0.615), (2.15, 0.715), (1.80, 0.805), (1.375, 0.865), (0.90, 0.905),
                               (0.45, 0.93), (-0.50, 0.95), (-1.40, 0.955), (-2.00, 0.945), (-2.30, 0.93),
                               (-2.48, 0.92)])
        self.p_fender = Profile([(2.30, 0.60), (2.15, 0.69), (1.80, 0.775), (1.375, 0.825), (0.90, 0.85),
                                 (0.40, 0.865), (0.00, 0.875), (-0.50, 0.89), (-1.00, 0.925), (-1.375, 0.95),
                                 (-1.90, 0.95), (-2.30, 0.935), (-2.48, 0.925)])
        self.p_width = Profile([(2.30, 0.80), (2.05, 0.94), (1.70, 0.968), (1.375, 0.975), (1.00, 0.965),
                                (0.40, 0.95), (-0.20, 0.945), (-0.80, 0.96), (-1.375, 0.975), (-1.90, 0.965),
                                (-2.30, 0.93), (-2.48, 0.90)])
        self.p_ywide = Profile([(2.30, 0.52), (1.375, 0.58), (0.0, 0.58), (-1.375, 0.62), (-2.48, 0.66)])
        self.p_floor = Profile([(2.30, 0.21), (2.0, 0.175), (1.2, 0.155), (-1.2, 0.155), (-1.9, 0.18), (-2.48, 0.27)])
        self.p_roof = Profile([(2.40, -0.30), (1.30, 0.25), (0.75, 0.72), (0.45, 0.905), (0.10, 1.12),
                               (-0.22, 1.265), (-0.55, 1.315), (-0.90, 1.31), (-1.30, 1.235), (-1.70, 1.10),
                               (-2.05, 0.99), (-2.30, 0.965), (-2.50, 0.985)])
        self.p_crown = Profile([(1.0, 0.36), (0.2, 0.32), (-0.4, 0.20), (-1.2, 0.20), (-2.0, 0.26), (-2.6, 0.30)])
        self.p_cwidth = Profile([(1.00, 0.50), (0.55, 0.70), (0.15, 0.775), (-0.40, 0.79), (-0.90, 0.775),
                                 (-1.40, 0.735), (-1.90, 0.68), (-2.30, 0.64), (-2.60, 0.62)])
        self.cabin = dict(z_front=0.36, z_rear=-1.15, y_floor=0.21, seat_x=0.36, seat_z=-0.52)

    def side_detail(self, Y, Z):
        win = sstep(0.24, 0.34, Y) * (1.0 - sstep(0.44, 0.62, Y))
        return -0.035 * win * window(Z, -0.80, 0.80, 0.18)

    GRILLE = sym_poly([(0.0, 0.275), (0.50, 0.275), (0.60, 0.40), (0.52, 0.585), (0.0, 0.585)])
    INTAKE = [(0.64, 0.255), (0.86, 0.275), (0.87, 0.40), (0.70, 0.40)]
    HEAD = [(0.585, 0.565), (0.90, 0.59), (0.915, 0.655), (0.60, 0.635)]
    HEAD_DRL = [(0.615, 0.622), (0.895, 0.642)]
    HEAD_PROJ = [(0.70, 0.598), (0.80, 0.606)]
    GILL = [(0.82, 0.62), (0.58, 0.62), (0.54, 0.47), (0.80, 0.47)]
    HVENT = [(0.30, 1.26), (0.50, 1.21), (0.50, 1.10), (0.30, 1.15)]
    TAIL = sym_poly([(0.0, 0.86), (0.80, 0.86), (0.86, 0.80), (0.80, 0.70), (0.0, 0.70)])
    HOOD_SEAM = sym_poly([(0.0, 2.21), (0.60, 2.13), (0.73, 1.60), (0.71, 0.64), (0.0, 0.57)])
    DOOR_SEAM = [(0.45, 0.275), (0.42, 0.70), (0.30, 1.05), (-0.62, 1.05), (-0.72, 0.70), (-0.68, 0.275)]
    TRUNK_SEAM = sym_poly([(0.0, -2.03), (0.55, -2.04), (0.60, -2.40), (0.0, -2.42)])

    def grille(self, X, Y, Z):
        return np.maximum(sdf.sd_polygon(X, Y, self.GRILLE) - 0.02, (self.front_face(np.abs(X), Y) - 0.16) - Z)

    def intake(self, X, Y, Z):
        aX = np.abs(X)
        return np.maximum(sdf.sd_polygon(aX, Y, self.INTAKE) - 0.02, (self.front_face(aX, Y) - 0.14) - Z)

    def tunnel(self, X, Y, Z):
        ramp = 0.155 + 0.20 * np.clip((-1.95 - Z) / 0.5, 0.0, 1.0)
        return smax(smax(np.abs(X) - 0.45, Y - ramp, 0.03), Z + 1.95, 0.03)

    def headlamp(self, X, Y, Z):
        return np.maximum(sdf.sd_polygon(np.abs(X), Y, self.HEAD) - 0.008, 1.6 - Z)

    def gill(self, X, Y, Z):
        return np.maximum(sdf.sd_polygon(Z, Y, self.GILL) - 0.01, 0.6 - np.abs(X))

    def hood_vent(self, X, Y, Z):
        return np.maximum(sdf.sd_polygon(np.abs(X), Z, self.HVENT) - 0.01, 0.6 - Y)

    def tail_panel(self, X, Y, Z):
        return np.maximum(sdf.sd_polygon(X, Y, self.TAIL) - 0.02, Z + 2.0)

    def cut_list(self):
        return [
            dict(fn=self.grille, k=0.02, mat='trim_black'),
            dict(fn=self.intake, k=0.02, mat='trim_black'),
            dict(fn=self.tunnel, k=0.02, mat='carbon'),
        ]

    PW = 0.032

    def _hts(self, P):
        return self.hull(P[:, 0], P[:, 1], P[:, 2]), self.canopy_ts(P[:, 0], P[:, 1], P[:, 2])

    def glass_ws(self, P):
        hull, ts = self._hts(P)
        return np.maximum(np.maximum(self.PW - ts, 0.055 - hull), -0.18 - P[:, 2])

    def glass_side(self, P):
        hull, ts = self._hts(P)
        return np.maximum(np.maximum(ts + self.PW, 0.045 - hull), (-1.48 + 1.5 * (P[:, 1] - 0.93)) - P[:, 2])

    def glass_rear(self, P):
        hull, ts = self._hts(P)
        return np.maximum(np.maximum(self.PW - ts, 0.05 - hull), np.maximum(P[:, 2] + 1.02, -1.98 - P[:, 2]))

    def glass_any(self, P):
        return np.minimum(np.minimum(self.glass_ws(P), self.glass_side(P)), self.glass_rear(P))

    def paint(self, skin, lod):
        self.paint_base(skin, lod)
        fine = lod == 0
        skin.region(lambda P: np.maximum(P[:, 1] - 0.245, np.abs(P[:, 2]) - 0.96), 'carbon', only=['paint'])
        skin.region(lambda P: np.maximum(P[:, 1] - 0.225, 1.80 - P[:, 2]), 'carbon', only=['paint'])
        skin.region(lambda P: np.maximum(P[:, 1] - 0.50, P[:, 2] + 1.83), 'carbon', only=['paint'])
        # glass in black frames, pillars and roof stay body colour
        for fld in (self.glass_ws, self.glass_side, self.glass_rear):
            skin.region(lambda P, f=fld: f(P) - 0.018, 'trim_black', only=['paint'], refine=0.03 if fine else None)
        for fld in (self.glass_ws, self.glass_side, self.glass_rear):
            skin.region(fld, 'glass', only=['trim_black'], refine=0.03 if fine else None)
        self.inlay(skin, self.headlamp, 0.016, fine)
        self.inlay(skin, self.tail_panel, 0.018, fine)
        if fine:
            self.inlay(skin, self.gill, 0.014, fine)
            self.inlay(skin, self.hood_vent, 0.014, fine)

    def decals(self, dec, lod):
        fine = lod == 0
        eps_l = 0.0015 if fine else 0.004
        step = 0.02 if fine else 0.06
        for sgn in (1, -1):
            pts = carlib.polyline_points([(sgn * x, y) for x, y in self.HEAD_DRL], step)
            dec.ribbon(dec.project_2d(pts, 'front'), 0.007, 'light_head', eps=eps_l, only=['trim_black'])
            for cx, cy in self.HEAD_PROJ:
                h = dec.project_2d([(sgn * cx, cy)], 'front')[0]
                if h is not None:
                    dec.disc(h[0], 0.021, 'light_head', eps=eps_l, segs=14 if fine else 8)
            # triple vertical tail bars: the inner one is the brake light
            for xb, mat in ((0.50, 'light_brake'), (0.62, 'light_tail'), (0.74, 'light_tail')):
                pts = carlib.polyline_points([(sgn * xb, 0.728), (sgn * xb, 0.835)], step)
                dec.ribbon(dec.project_2d(pts, 'rear'), 0.022, mat, eps=eps_l, only=['trim_black'])
        xs = carlib.polyline_points([(-0.20, -1.045), (0.20, -1.045)], step)
        dec.ribbon(dec.project_2d(xs, 'top'), 0.008, 'light_brake', eps=eps_l)
        if not fine:
            return
        # grille bars on the back wall of the grille opening
        for yb in (0.35, 0.43, 0.51):
            xs = carlib.polyline_points([(-0.47, yb), (0.47, yb)], 0.05)
            dec.ribbon(dec.project_2d(xs, 'front'), 0.007, 'metal', eps=0.002, only=['trim_black'])
        hood = carlib.outline_points(self.HOOD_SEAM, round_r=0.03, step=0.025)
        dec.ribbon(dec.project_2d(hood, 'top'), 0.003, 'trim_black', eps=0.001, only=['paint'], closed=True)
        trunk = carlib.outline_points(self.TRUNK_SEAM, round_r=0.03, step=0.025)
        dec.ribbon(dec.project_2d(trunk, 'top'), 0.003, 'trim_black', eps=0.001, only=['paint'], closed=True)
        door = carlib.outline_points(self.DOOR_SEAM, round_r=0.04, step=0.025)
        for sgn in (1, -1):
            dec.ribbon(dec.project_2d(door, 'side', sgn), 0.003, 'trim_black', eps=0.001, only=['paint'], closed=True)
        # fuel filler door (left rear quarter)
        ang = np.linspace(0, 2 * np.pi, 28, endpoint=False)
        cap = np.stack([-1.80 + 0.075 * np.cos(ang), 0.80 + 0.075 * np.sin(ang)], axis=1)
        dec.ribbon(dec.project_2d(cap, 'side', 1), 0.0025, 'trim_black', eps=0.001, only=['paint'], closed=True)

    def parts(self, lod):
        out = []
        p, fn = carlib.part_splitter(self, 0.135, 0.172, ext=0.03, z_back=1.85, lod=lod)
        out.append(p); self.part_fields.append(fn)
        p, fn = carlib.part_skirt(self, -0.95, 0.95, 0.125, 0.160, lod=lod)
        out.append(p); self.part_fields.append(fn)
        ramp = lambda Z: 0.155 + 0.20 * np.clip((-1.95 - Z) / 0.5, 0.0, 1.0)
        p, fn = carlib.part_fins(self, [0.0, 0.15, 0.30], -2.0, -2.43, 0.14, 0.19, top_fn=ramp, tris=600, lod=lod)
        out.append(p); self.part_fields.append(fn)
        ps, fn = carlib.part_mirrors(np.array([0.985, 0.965, 0.36]), (0.09, 0.055, 0.078),
                                     np.array([0.875, 0.895, 0.39]), np.array([0.935, 0.945, 0.37]), lod=lod)
        out += ps; self.part_fields.append(fn)
        for xe in (0.49, 0.635):
            out += carlib.part_exhaust(xe, 0.345, -2.365, length=0.12, r_out=0.058, r_in=0.050, lod=lod)
        return out

    def interior(self, lod):
        c = self.cabin
        return carlib.build_interior(self, c['z_front'], c['z_rear'], c['y_floor'], c['seat_x'], c['seat_z'],
                                     self.glass_any, lod=lod)

    def points(self):
        c = self.cabin
        return {
            'headlightL': (0.75, 0.61, 2.20), 'headlightR': (-0.75, 0.61, 2.20),
            'taillightL': (0.62, 0.78, -2.43), 'taillightR': (-0.62, 0.78, -2.43),
            'exhaustL': (0.57, 0.335, -2.41), 'exhaustR': (-0.57, 0.335, -2.41),
            'camHood': (0.0, 1.03, 0.55), 'camCockpit': (c['seat_x'], 1.09, c['seat_z'] - 0.06),
            'camBumper': (0.0, 0.50, 2.33), 'wingCenter': (0.0, 0.99, -2.42),
        }


class Zephyr(CarBase):
    """ZEPHYR RX - compact lightweight track weapon: short overhangs, box flares, big splitter with
    canards, swan-neck rear wing, roof scoop, twin stripes."""
    id = 'zephyr'
    name = 'ZEPHYR RX'
    paint_color = (0.80, 0.50, 0.02)

    def setup(self):
        self.axle_f, self.axle_r = 1.26, -1.26
        self.track_f, self.track_r = 1.60, 1.58
        self.rf, self.rr = 0.325, 0.335
        self.wf, self.wr = 0.245, 0.285
        self.gap_f, self.gap_r = 0.045, 0.05
        self.half_w = 0.93
        self.z_nose, self.z_tail = 2.09, -2.16
        self.height = 1.34
        self.rim_style = 'multi10'
        self.concave_f, self.concave_r = 0.03, 0.05
        self.k_shoulder, self.k_nose, self.k_tail, self.k_sill = 0.045, 0.05, 0.04, 0.03
        self.lean_up = 0.06
        self.arch_lip, self.arch_lip_off, self.arch_lip_w = 0.020, 0.040, 0.034
        self.nose_plan, self.nose_pow, self.nose_vert, self.nose_tip_y = 0.40, 2.2, 0.8, 0.30
        self.tail_plan, self.tail_pow, self.tail_tuck, self.tail_tuck_y = 0.22, 3.0, 0.07, (0.50, 0.26)
        self.tumble, self.belt_ref, self.canopy_drop = 0.42, 0.82, 0.07
        self.p_hood = Profile([(2.09, 0.40), (1.95, 0.485), (1.60, 0.61), (1.26, 0.695), (0.95, 0.755),
                               (0.70, 0.795), (0.20, 0.85), (-0.60, 0.90), (-1.30, 0.93), (-1.90, 0.94),
                               (-2.16, 0.935)])
        self.p_fender = Profile([(2.09, 0.40), (1.96, 0.54), (1.68, 0.715), (1.26, 0.775), (0.90, 0.775),
                                 (0.40, 0.785), (0.00, 0.805), (-0.50, 0.845), (-0.90, 0.905), (-1.26, 0.94),
                                 (-1.70, 0.945), (-2.00, 0.935), (-2.16, 0.925)])
        self.p_width = Profile([(2.09, 0.66), (1.90, 0.86), (1.60, 0.915), (1.26, 0.93), (0.90, 0.915),
                                (0.40, 0.885), (0.00, 0.875), (-0.40, 0.89), (-0.85, 0.925), (-1.26, 0.935),
                                (-1.65, 0.925), (-1.95, 0.90), (-2.16, 0.875)])
        self.p_ywide = Profile([(2.09, 0.38), (1.26, 0.48), (0.0, 0.50), (-1.26, 0.56), (-2.16, 0.60)])
        self.p_floor = Profile([(2.09, 0.12), (1.8, 0.105), (1.0, 0.10), (-1.0, 0.10), (-1.6, 0.115), (-2.16, 0.19)])
        self.p_roof = Profile([(2.20, -0.30), (1.50, 0.25), (1.10, 0.60), (0.85, 0.78), (0.55, 1.00),
                               (0.25, 1.18), (-0.05, 1.255), (-0.40, 1.26), (-0.80, 1.20), (-1.20, 1.07),
                               (-1.60, 0.965), (-2.00, 0.905), (-2.30, 0.87)])
        self.p_crown = Profile([(1.3, 0.38), (0.4, 0.34), (-0.1, 0.24), (-1.0, 0.24), (-2.3, 0.32)])
        self.p_cwidth = Profile([(1.30, 0.48), (0.90, 0.64), (0.50, 0.725), (0.10, 0.755), (-0.40, 0.75),
                                 (-0.90, 0.70), (-1.40, 0.60), (-1.90, 0.50), (-2.30, 0.44)])
        self.cabin = dict(z_front=0.74, z_rear=-0.60, y_floor=0.155, seat_x=0.35, seat_z=-0.12)

    def side_detail(self, Y, Z):
        y_hi = 0.50 + 0.14 * sstep(0.80, -0.70, Z)
        win = sstep(0.20, 0.38, Y) * (1.0 - sstep(y_hi - 0.06, y_hi + 0.06, Y))
        return -0.075 * sstep(0.78, -0.40, Z) * (1.0 - sstep(-0.62, -0.72, Z)) * win

    def scoop(self, X, Y, Z):
        """Roof air scoop."""
        yc = self.p_roof(Z) - 0.020
        return sdf.sd_box(np.abs(X), Y - yc, Z, (0.0, 0.0, -0.24), (0.105, 0.062, 0.37), 0.05)

    def add_volumes(self, f, X, Y, Z):
        return smin(f, self.scoop(X, Y, Z), 0.035)

    def scoop_mouth(self, X, Y, Z):
        yc = self.p_roof(0.10) + 0.006
        return sdf.sd_box(np.abs(X), Y, Z, (0.0, float(yc), 0.17), (0.068, 0.022, 0.12), 0.016)

    INTAKE = [(0.0, 0.155), (0.80, 0.175), (0.83, 0.285), (0.70, 0.345), (0.0, 0.33)]
    HEAD = [(0.38, 2.02), (0.60, 1.945), (0.78, 1.80), (0.855, 1.60), (0.83, 1.545), (0.74, 1.64), (0.62, 1.79), (0.40, 1.94)]
    HEAD_DRL = [(0.405, 1.975), (0.60, 1.875), (0.735, 1.73), (0.825, 1.585)]
    HEAD_PROJ = [(0.66, 1.845), (0.745, 1.755)]
    HOOD_DUCT = sym_poly([(0.0, 1.50), (0.24, 1.46), (0.30, 1.18), (0.0, 1.14)])
    FENDER_VENT = [(0.95, 0.66), (0.80, 0.66), (0.78, 0.45), (0.93, 0.45)]
    TAIL = sym_poly([(0.0, 0.875), (0.80, 0.875), (0.85, 0.80), (0.80, 0.715), (0.0, 0.715)])
    HOOD_SEAM = sym_poly([(0.0, 1.95), (0.34, 1.89), (0.58, 1.55), (0.62, 1.05), (0.48, 0.92), (0.0, 0.94)])
    DOOR_SEAM = [(0.74, 0.215), (0.72, 0.58), (0.62, 0.95), (-0.46, 0.95), (-0.52, 0.58), (-0.48, 0.215)]

    def intake(self, X, Y, Z):
        aX = np.abs(X)
        return np.maximum(sdf.sd_polygon(X, Y, sym_poly(self.INTAKE)) - 0.022, (self.front_face(aX, Y) - 0.20) - Z)

    def duct(self, X, Y, Z):
        return sdf.sd_box(np.abs(X), Y, Z, (0.762, 0.47, -0.735), (0.062, 0.12, 0.095), 0.035)

    def tunnel(self, X, Y, Z):
        ramp = 0.10 + 0.30 * np.clip((-1.50 - Z) / 0.66, 0.0, 1.0)
        return smax(smax(np.abs(X) - 0.68, Y - ramp, 0.03), Z + 1.50, 0.03)

    def headlamp(self, X, Y, Z):
        return np.maximum(sdf.sd_polygon(np.abs(X), Z, self.HEAD) - 0.010, 0.40 - Y)

    def hood_duct(self, X, Y, Z):
        return np.maximum(sdf.sd_polygon(X, Z, self.HOOD_DUCT) - 0.015, 0.5 - Y)

    def fender_vent(self, X, Y, Z):
        return np.maximum(sdf.sd_polygon(Z, Y, self.FENDER_VENT) - 0.01, 0.6 - np.abs(X))

    def tail_panel(self, X, Y, Z):
        return np.maximum(sdf.sd_polygon(X, Y, self.TAIL) - 0.02, Z + 1.75)

    def cut_list(self):
        return [
            dict(fn=self.duct, k=0.015, mat='trim_black'),
            dict(fn=self.tunnel, k=0.02, mat='carbon'),
            dict(fn=self.intake, k=0.02, mat='trim_black'),
            dict(fn=self.scoop_mouth, k=0.012, mat='trim_black'),
        ]

    PW = 0.030

    def _hts(self, P):
        return self.hull(P[:, 0], P[:, 1], P[:, 2]), self.canopy_ts(P[:, 0], P[:, 1], P[:, 2])

    def glass_ws(self, P):
        hull, ts = self._hts(P)
        return np.maximum(np.maximum(self.PW - ts, 0.055 - hull), 0.24 - P[:, 2])

    def glass_side(self, P):
        hull, ts = self._hts(P)
        return np.maximum(np.maximum(ts + self.PW, 0.045 - hull), (-0.52 + 0.55 * (P[:, 1] - 0.86)) - P[:, 2])

    def glass_rear(self, P):
        hull, ts = self._hts(P)
        sc = self.scoop(P[:, 0], P[:, 1], P[:, 2])
        return np.maximum(np.maximum(np.maximum(self.PW - ts, 0.05 - hull), np.maximum(P[:, 2] + 0.84, -1.24 - P[:, 2])), 0.03 - sc)

    def glass_any(self, P):
        return np.minimum(np.minimum(self.glass_ws(P), self.glass_side(P)), self.glass_rear(P))

    def paint(self, skin, lod):
        self.paint_base(skin, lod)
        fine = lod == 0
        hull = PF(self.hull)
        ts = PF(self.canopy_ts)
        skin.region(lambda P: np.maximum(P[:, 1] - 0.19, np.abs(P[:, 2]) - 0.885), 'carbon', only=['paint'])
        skin.region(lambda P: np.maximum(P[:, 1] - 0.155, 1.62 - P[:, 2]), 'carbon', only=['paint'])
        skin.region(lambda P: np.maximum(P[:, 1] - 0.42, P[:, 2] + 1.68), 'carbon', only=['paint'])
        # twin stripes (accent colour) over nose, hood and deck
        skin.band(lambda P: np.abs(P[:, 0]) - 0.085, 0.05, 'paint_accent', only=['paint'],
                  extent=lambda P: 0.36 - P[:, 1])
        # greenhouse: black pillars, carbon roof with the scoop, body-colour tail
        gh = self.greenhouse
        skin.region(gh, 'trim_black', only=['paint', 'paint_accent'], refine=0.03 if fine else None)
        skin.region(lambda P: np.maximum(gh(P), P[:, 2] + 1.30), 'paint', only=['trim_black'])
        sail = lambda P: np.maximum(np.maximum(gh(P), ts(P) + self.PW), P[:, 2] - (-0.55 + 0.55 * (P[:, 1] - 0.86)))
        skin.region(sail, 'paint', only=['trim_black'], refine=0.03 if fine else None)
        roof = lambda P: np.maximum(np.maximum(self.PW - ts(P), P[:, 2] - 0.21), -0.81 - P[:, 2])
        skin.region(roof, 'carbon', only=['trim_black'], refine=0.03 if fine else None)
        sc = PF(self.scoop)
        skin.region(lambda P: sc(P) - 0.012, 'carbon', only=['trim_black', 'paint', 'paint_accent'])
        for fld in (self.glass_ws, self.glass_side, self.glass_rear):
            skin.region(fld, 'glass', only=['trim_black'], refine=0.03 if fine else None)
        self.inlay(skin, self.headlamp, 0.016, fine)
        self.inlay(skin, self.tail_panel, 0.02, fine, only=('paint', 'paint_accent'))
        if fine:
            self.inlay(skin, self.hood_duct, 0.03, fine, only=('paint', 'paint_accent'), ramp=0.02)
            self.inlay(skin, self.fender_vent, 0.014, fine)

    def decals(self, dec, lod):
        fine = lod == 0
        eps_l = 0.0015 if fine else 0.004
        step = 0.02 if fine else 0.06
        ang = np.linspace(0, 2 * np.pi, 26 if fine else 10, endpoint=False)
        for sgn in (1, -1):
            pts = carlib.polyline_points([(sgn * x, z) for x, z in self.HEAD_DRL], step)
            dec.ribbon(dec.project_2d(pts, 'top'), 0.008, 'light_head', eps=eps_l, only=['trim_black'])
            for cx, cz in self.HEAD_PROJ:
                h = dec.project_2d([(sgn * cx, cz)], 'top')[0]
                if h is not None:
                    dec.disc(h[0], 0.022, 'light_head', eps=eps_l, segs=14 if fine else 8)
            # twin afterburner rings per side: ring = tail light, core = brake light
            for cx in (0.47, 0.67):
                ring = np.stack([sgn * cx + 0.058 * np.cos(ang), 0.795 + 0.058 * np.sin(ang)], axis=1)
                dec.ribbon(dec.project_2d(ring, 'rear'), 0.010, 'light_tail', eps=eps_l, only=['trim_black'], closed=True)
                h = dec.project_2d([(sgn * cx, 0.795)], 'rear')[0]
                if h is not None:
                    dec.disc(h[0], 0.028, 'light_brake', eps=eps_l, segs=14 if fine else 8)
        if not fine:
            return
        hood = carlib.outline_points(self.HOOD_SEAM, round_r=0.03, step=0.025)
        dec.ribbon(dec.project_2d(hood, 'top'), 0.003, 'trim_black', eps=0.001, only=['paint', 'paint_accent'], closed=True)
        door = carlib.outline_points(self.DOOR_SEAM, round_r=0.04, step=0.025)
        for sgn in (1, -1):
            dec.ribbon(dec.project_2d(door, 'side', sgn), 0.003, 'trim_black', eps=0.001, only=['paint'], closed=True)

    def parts(self, lod):
        out = []
        p, fn = carlib.part_splitter(self, 0.082, 0.118, ext=0.075, z_back=1.55, lod=lod)
        out.append(p); self.part_fields.append(fn)
        p, fn = carlib.part_skirt(self, -0.86, 0.88, 0.072, 0.105, out=0.03, lod=lod)
        out.append(p); self.part_fields.append(fn)
        ramp = lambda Z: 0.10 + 0.30 * np.clip((-1.50 - Z) / 0.66, 0.0, 1.0)
        p, fn = carlib.part_fins(self, [0.0, 0.17, 0.34, 0.51], -1.56, -2.13, 0.09, 0.16, top_fn=ramp, lod=lod)
        out.append(p); self.part_fields.append(fn)
        p, fn = carlib.part_wing(zc=-2.02, yc=1.285, chord=0.33, span_half=0.83, thick=0.03, angle_deg=10.0,
                                 pylons=(0.36,), deck_y=0.935, endplate=0.085, pylon_len=0.05, swan=True, tris=2200, lod=lod)
        out.append(p); self.part_fields.append(fn)
        # dive planes (canards) on the bumper corners
        def canard(X, Y, Z):
            aX = np.abs(X)
            d = 1e9
            for yc, zc in ((0.37, 1.70), (0.275, 1.74)):
                dz = Z - zc; dy = Y - yc
                a = np.radians(14.0)
                v = dz * np.cos(a) - dy * np.sin(a)
                w = dz * np.sin(a) + dy * np.cos(a)
                d = np.minimum(d, sdf.sd_box(aX, w, v, (0.845, 0.0, 0.0), (0.075, 0.006, 0.075), 0.005))
            return d
        out.append(sdf_part(canard, (-0.96, 0.2, 1.58), (0.96, 0.46, 1.86), 0.004 if lod == 0 else 0.01, 'carbon',
                            target_tris=360 if lod == 0 else 80, normals='auto'))
        self.part_fields.append(canard)
        ps, fn = carlib.part_mirrors(np.array([0.93, 0.895, 0.56]), (0.082, 0.05, 0.07),
                                     np.array([0.835, 0.825, 0.585]), np.array([0.885, 0.875, 0.57]), lod=lod, pod_mat='paint_accent')
        out += ps; self.part_fields.append(fn)
        out += carlib.part_exhaust(0.085, 0.60, -2.195, length=0.14, r_out=0.062, r_in=0.053, lod=lod)
        return out

    def interior(self, lod):
        c = self.cabin
        return carlib.build_interior(self, c['z_front'], c['z_rear'], c['y_floor'], c['seat_x'], c['seat_z'],
                                     self.glass_any, lod=lod)

    def points(self):
        c = self.cabin
        return {
            'headlightL': (0.68, 0.57, 1.82), 'headlightR': (-0.68, 0.57, 1.82),
            'taillightL': (0.57, 0.795, -2.13), 'taillightR': (-0.57, 0.795, -2.13),
            'exhaustL': (0.085, 0.60, -2.16), 'exhaustR': (-0.085, 0.60, -2.16),
            'camHood': (0.0, 0.94, 0.95), 'camCockpit': (c['seat_x'], 1.02, c['seat_z'] - 0.06),
            'camBumper': (0.0, 0.40, 2.12), 'wingCenter': (0.0, 1.285, -2.02),
        }


CARS = {'aurion': Aurion, 'valken': Valken, 'zephyr': Zephyr}
