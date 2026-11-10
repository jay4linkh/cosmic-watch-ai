"""Sun detector: solar coordinates, limb fit, spots found where they were painted, NOAA matching."""
import cv2
import numpy as np

from cosmic_watch.detectors import sun


def test_ephemeris_known_values():
    e = sun.ephemeris("2026-11-10 07:10:06")
    assert abs(e["b0_deg"] - 3.42) < 0.1                     # early November: the north pole tips slightly toward us
    assert abs(e["radius_arcsec"] - 969.0) < 1.0
    assert abs(sun.ephemeris("2026-06-06 12:00:00")["b0_deg"]) < 0.3      # B0 crosses zero around 6 June
    assert sun.ephemeris("2026-03-06 12:00:00")["b0_deg"] < -7.0          # and is most negative in early March


def test_heliographic_centre_and_limb():
    lat, lon = sun.heliographic(np.array(0.0), np.array(0.0), 5.0)
    assert abs(lat - 5.0) < 1e-6 and abs(lon) < 1e-6                      # disk centre sits at latitude B0
    lat, lon = sun.heliographic(np.array(0.5), np.array(0.0), 0.0)
    assert abs(lat) < 1e-6 and abs(lon - 30.0) < 1e-6                     # half a radius west = 30 deg west


def fake_sun(spots, r=480.0, size=1024):
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    c = (size - 1) / 2
    rho = np.hypot(xx - c, yy - c) / r
    mu = np.sqrt(np.clip(1 - rho ** 2, 0, 1))
    img = np.where(rho < 1, 200 * (1 - 0.35 * (1 - mu) - 0.1 * (1 - mu) ** 2), 0).astype(np.float32)
    for lat, lon, rad in spots:
        la, lo = np.radians(lat), np.radians(lon)
        x, y = c + r * np.cos(la) * np.sin(lo), c - r * np.sin(la)
        img[np.hypot(xx - x, yy - y) < rad] *= 0.45
    return cv2.cvtColor(cv2.GaussianBlur(img, (0, 0), 0.8).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def test_spots_groups_and_noaa(tmp_path):
    p = tmp_path / "sun.png"
    cv2.imwrite(str(p), fake_sun([(10, -30, 6), (11, -27, 4), (-15, 40, 7), (5, 70, 5)]))
    when = "2026-06-06 12:00:00"                                           # B0 ~ 0, so painted and measured latitudes agree
    noaa = {"date": "2026-06-05", "regions": [{"region": 1001, "location": "N10E35", "spots": 2, "area": 100},
                                               {"region": 1002, "location": "S15W33", "spots": 1, "area": 120},
                                               {"region": 1003, "location": "S40E10", "spots": 1, "area": 10}]}
    _, spots, st = sun.sunspots(p, when, noaa)
    assert abs(st["disk_radius_px"] - 480) < 2 and abs(st["disk_center_px"][0] - 511.5) < 1.5
    assert st["n_sunspots"] == 4 and st["n_groups"] == 3 and st["sunspot_number"] == 34
    big = max(st["groups"], key=lambda g: g["n_spots"])
    assert abs(big["lat"] - 10.4) < 1.5 and abs(big["lon"] + 29) < 2.5
    # the spots are painted as flat circles on the image, so the one near the limb really covers more of the surface:
    # (5 px)^2 / mu(70 deg) against (7 px)^2 / mu(40 deg) = 1.13. Without the foreshortening correction it would be 0.51.
    limb = [s for s in spots if s["lon"] > 60][0]
    centre = [s for s in spots if 30 < s["lon"] < 50][0]
    assert 0.9 < limb["area_msh"] / centre["area_msh"] < 1.45
    # the list is dated the 5th, i.e. valid at 24:00 UT that day; the image is 12 h later, so regions sit ~7 deg further west
    assert st["noaa"]["noaa_recovered"] == 2 and st["noaa"]["missed"] == [1003]
    assert {g.get("noaa") for g in st["groups"]} >= {1001, 1002}


def test_quiet_sun_has_no_spots(tmp_path):
    p = tmp_path / "quiet.png"
    cv2.imwrite(str(p), fake_sun([]))
    _, spots, st = sun.sunspots(p, "2026-06-06 12:00:00")
    assert st["n_sunspots"] == 0 and st["sunspot_number"] == 0
