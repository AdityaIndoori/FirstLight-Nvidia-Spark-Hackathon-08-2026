"""Flight plan exports for what government teams actually fly.

Six formats from one GeoJSON FeatureCollection. No internet, no SDKs: these are
all text formats and the whole point is that a crew can leave with a file.
"""
from __future__ import annotations

import json
from typing import Any

MIME = {
    "plan": ("application/json", "firstlight-survey.plan"),
    "waypoints": ("text/plain", "firstlight-survey.waypoints"),
    "kml": ("application/vnd.google-earth.kml+xml", "firstlight-survey.kml"),
    "litchi": ("text/csv", "firstlight-survey-litchi.csv"),
    "gpx": ("application/gpx+xml", "firstlight-survey.gpx"),
    "geojson": ("application/geo+json", "firstlight-survey.geojson"),
}


def _path_and_props(fc: dict) -> tuple[list[list[float]], dict]:
    for f in fc.get("features", []):
        if (f.get("properties") or {}).get("role") == "survey-path":
            return list(f["geometry"]["coordinates"]), dict(f.get("properties") or {})
    raise ValueError("no survey-path feature in the flight plan")


def render(fc: dict, fmt: str) -> tuple[bytes, str, str]:
    fmt = (fmt or "plan").lower()
    if fmt not in MIME:
        raise ValueError(f"unknown format {fmt!r}, expected one of {sorted(MIME)}")
    mime, name = MIME[fmt]
    body = _RENDERERS[fmt](fc)
    return body.encode("utf-8"), mime, name


def _geojson(fc: dict) -> str:
    return json.dumps(fc, indent=2)


def _qgc_plan(fc: dict) -> str:
    """QGroundControl .plan, PX4 and ArduPilot. Item 1 is takeoff, then waypoints,
    then RTL, because a plan that cannot launch or recover is not flyable."""
    coords, props = _path_and_props(fc)
    alt = float(props.get("altitude_m_agl", 90))
    items: list[dict[str, Any]] = []
    seq = 1
    first = coords[0] if coords else [0.0, 0.0]
    items.append(_mav(seq, 22, [0, 0, 0, 0, first[1], first[0], alt]))  # NAV_TAKEOFF
    for lng, lat in coords:
        seq += 1
        items.append(_mav(seq, 16, [0, 0, 0, 0, lat, lng, alt]))  # NAV_WAYPOINT
    seq += 1
    items.append(_mav(seq, 20, [0, 0, 0, 0, 0, 0, 0]))  # NAV_RETURN_TO_LAUNCH
    return json.dumps(
        {
            "fileType": "Plan",
            "version": 1,
            "groundStation": "FIRST LIGHT",
            "geoFence": {"circles": [], "polygons": [], "version": 2},
            "rallyPoints": {"points": [], "version": 2},
            "mission": {
                "version": 2,
                "firmwareType": 12,  # PX4
                "vehicleType": 2,  # multirotor
                "cruiseSpeed": 10,
                "hoverSpeed": 5,
                "globalPlanAltitudeMode": 1,
                "plannedHomePosition": [first[1], first[0], 0],
                "items": items,
            },
        },
        indent=2,
    )


def _mav(seq: int, command: int, params: list[float]) -> dict:
    return {
        "AMSLAltAboveTerrain": None,
        "Altitude": params[6],
        "AltitudeMode": 1,
        "autoContinue": True,
        "command": command,
        "doJumpId": seq,
        "frame": 3,
        "params": params,
        "type": "SimpleItem",
    }


def _waypoints(fc: dict) -> str:
    """Mission Planner .waypoints, tab separated MAVLink."""
    coords, props = _path_and_props(fc)
    alt = float(props.get("altitude_m_agl", 90))
    lines = ["QGC WPL 110"]
    for i, (lng, lat) in enumerate(coords):
        current = 1 if i == 0 else 0
        lines.append(
            f"{i}\t{current}\t3\t16\t0\t0\t0\t0\t{lat:.7f}\t{lng:.7f}\t{alt:.1f}\t1"
        )
    return "\n".join(lines) + "\n"


def _kml(fc: dict) -> str:
    """KML for DJI Pilot 2 and Google Earth."""
    coords, props = _path_and_props(fc)
    alt = float(props.get("altitude_m_agl", 90))
    pts = " ".join(f"{lng:.7f},{lat:.7f},{alt:.0f}" for lng, lat in coords)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>\n'
        "<name>FIRST LIGHT survey</name>\n"
        "<Placemark><name>survey path</name>\n"
        "<LineString><altitudeMode>relativeToGround</altitudeMode>\n"
        f"<coordinates>{pts}</coordinates></LineString></Placemark>\n"
        "</Document></kml>\n"
    )


def _litchi(fc: dict) -> str:
    """Litchi CSV for DJI consumer airframes."""
    coords, props = _path_and_props(fc)
    alt = float(props.get("altitude_m_agl", 90))
    head = (
        "latitude,longitude,altitude(m),heading(deg),curvesize(m),rotationdir,"
        "gimbalmode,gimbalpitchangle,actiontype1,actionparam1,altitudemode,speed(m/s)"
    )
    rows = [head]
    for lng, lat in coords:
        rows.append(f"{lat:.7f},{lng:.7f},{alt:.1f},0,0,0,2,-90,-1,0,1,8")
    return "\n".join(rows) + "\n"


def _gpx(fc: dict) -> str:
    coords, props = _path_and_props(fc)
    alt = float(props.get("altitude_m_agl", 90))
    pts = "\n".join(
        f'    <trkpt lat="{lat:.7f}" lon="{lng:.7f}"><ele>{alt:.0f}</ele></trkpt>'
        for lng, lat in coords
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<gpx version="1.1" creator="FIRST LIGHT" xmlns="http://www.topografix.com/GPX/1/1">\n'
        "  <trk><name>FIRST LIGHT survey</name><trkseg>\n"
        f"{pts}\n"
        "  </trkseg></trk>\n</gpx>\n"
    )


_RENDERERS = {
    "plan": _qgc_plan,
    "waypoints": _waypoints,
    "kml": _kml,
    "litchi": _litchi,
    "gpx": _gpx,
    "geojson": _geojson,
}
