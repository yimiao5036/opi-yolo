import math

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
    # WGS-84 椭球参数（赤道半径，单位：米）
    EARTH_RADIUS = 6378137.0
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