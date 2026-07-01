import math

# WGS-84 赤道半径（米）
EARTH_RADIUS = 6378137.0


def latlon_to_ned(lat, lon, alt, ref_lat, ref_lon, ref_alt):
    """
    将 WGS-84 经纬高 (LLA) 转换为局部 NED 坐标 (x, y, z)

    Args:
        lat, lon, alt:   目标点的纬度(deg)、经度(deg)、海拔(m)
        ref_lat, ref_lon, ref_alt: 参考点（起飞点）的纬度、经度、海拔

    Returns:
        (x, y, z): NED 坐标 (米)
            x: 北向距离 (正北为正)
            y: 东向距离 (正东为正)
            z: 向下距离 (向下为正，即向上为负)
    """
    # 1. 纬度差 -> 北向距离 (x)
    dlat = math.radians(lat - ref_lat)
    x = dlat * EARTH_RADIUS

    # 2. 经度差 -> 东向距离 (y)，需乘以纬度的余弦值修正
    dlon = math.radians(lon - ref_lon)
    y = dlon * EARTH_RADIUS * math.cos(math.radians(ref_lat))

    # 3. 高度差 -> NED Z (向下为正)
    # 若目标海拔高于参考点，则 z 为负（向上）
    z = -(alt - ref_alt)

    return x, y, z


def latlon_alt_to_local_offset(target_lat, target_lon, target_alt,
                               current_lat, current_lon, current_alt):
    """
    将目标经纬高转换为相对当前位置的 LOCAL_OFFSET_NED 偏移量。

    Args:
        target_lat, target_lon: 目标纬度/经度（度）
        target_alt:             目标相对高度（米，正值向上）
        current_lat, current_lon: 当前纬度/经度（度）
        current_alt:              当前相对高度（米，正值向上）

    Returns:
        (x, y, z): 相对当前位置的 NED 偏移（米）
            x: 北向偏移，正值向北
            y: 东向偏移，正值向东
            z: 下向偏移，正值下降，负值爬升
    """
    return latlon_to_ned(
        target_lat, target_lon, target_alt,
        current_lat, current_lon, current_alt,
    )


def haversine(lat1, lon1, lat2, lon2):
    """
    使用 Haversine 公式计算两个经纬度点之间的球面距离（米）

    Args:
        lat1, lon1: 起点纬度、经度（度）
        lat2, lon2: 终点纬度、经度（度）

    Returns:
        水平距离（米）
    """
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS * c


def dist_3d_latlon(lat1, lon1, alt1, lat2, lon2, alt2):
    """
    计算两个经纬高坐标之间的 3D 距离（米）

    水平距离使用 Haversine 公式，垂直距离使用高度差。
    此处的 alt 为相对高度（正值向上），与 SETPOINT 协议一致。

    Args:
        lat1, lon1, alt1: 起点纬度(度)、经度(度)、相对高度(米)
        lat2, lon2, alt2: 终点纬度(度)、经度(度)、相对高度(米)

    Returns:
        3D 距离（米）
    """
    horizontal = haversine(lat1, lon1, lat2, lon2)
    vertical = alt1 - alt2
    return math.sqrt(horizontal ** 2 + vertical ** 2)
