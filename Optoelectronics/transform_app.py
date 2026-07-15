import socket
import struct
import json
import threading
import time
import zlib
import hashlib
import hmac

# 固定常量
HEADER_MAGIC = 0xA5A5A5A5  # 消息头标记
# < 表示小端字节序；I=4字节，B=1字节
# 字段顺序: magic(4B), msg_id(4B), msg_type(1B), sub_pack(1B), res1(1B), res2(1B), data_len(4B)
STRUCT_FORMAT = '<IIBBBB I'


def pack_message(msg_id: int, json_data: dict) -> bytes:
    """
    将 Python 字典打包成符合光电 IPC 协议的二进制 TCP 报文
    """
    json_bytes = json.dumps(json_data).encode('utf-8')
    data_len = len(json_bytes)

    # 组装 16 字节头部
    msg_type = 1  # 1：JSON格式文本数据
    sub_pack = 0  # 0表示不分包
    res1 = 0
    res2 = 0

    header_bytes = struct.pack(
        STRUCT_FORMAT,
        HEADER_MAGIC,
        msg_id,
        msg_type,
        sub_pack,
        res1,
        res2,
        data_len
    )

    # 计算从消息头开始到消息内容结束的 CRC32 校验码
    packet_payload = header_bytes + json_bytes
    crc32_val = zlib.crc32(packet_payload) & 0xffffffff
    crc_bytes = struct.pack('<I', crc32_val)  # 4字节校验码

    return packet_payload + crc_bytes


def unpack_message(sock) -> tuple[int, dict] | None:
    """
    从 socket 阻塞读取并解析一条完整的 TCP 报文
    """
    try:
        # 1. 先读取 16 字节的固定头部
        header_bytes = b""
        while len(header_bytes) < 16:
            chunk = sock.recv(16 - len(header_bytes))
            if not chunk:
                return None  # 连接断开
            header_bytes += chunk

        # 2. 解析头部
        magic, msg_id, msg_type, sub_pack, res1, res2, data_len = struct.unpack(STRUCT_FORMAT, header_bytes)

        if magic != HEADER_MAGIC:
            raise ValueError(f"不合法的消息头标记: {hex(magic)}")

        # 3. 根据头部指明的数据长度，完整读取 JSON 文本
        json_bytes = b""
        while len(json_bytes) < data_len:
            chunk = sock.recv(data_len - len(json_bytes))
            if not chunk:
                return None
            json_bytes += chunk

        # 4. 读取 4 字节的 CRC32 校验码
        crc_bytes = b""
        while len(crc_bytes) < 4:
            chunk = sock.recv(4 - len(crc_bytes))
            if not chunk:
                return None
            crc_bytes += chunk

        # 5. 校验数值
        received_crc, = struct.unpack('<I', crc_bytes)
        local_crc = zlib.crc32(header_bytes + json_bytes) & 0xffffffff
        if received_crc != local_crc:
            print(f"[警告] CRC校验失败! 丢弃该包。收到: {received_crc}, 计算: {local_crc}")
            return None

        # 6. 解析出字典格式的数据
        json_data = json.loads(json_bytes.decode('utf-8'))
        return msg_id, json_data
    except Exception as e:
        print(f"[网络] 读取数据时发生异常: {e}")
        return None


def heartbeat_thread(sock, token_dict):
    """每15秒发一次心跳包的子线程"""
    while True:
        time.sleep(15.0)
        if 'token' in token_dict:
            heartbeat_msg = {
                "cmd": "userOnlineHeart",
                "param": {"token": token_dict['token']}
            }
            # 【修复 Bug 1】：传入正确的 msg_id (1003) 和内容
            packet = pack_message(1003, heartbeat_msg)
            try:
                sock.sendall(packet)
                print("[心跳] 已发送保活包")
            except socket.error:
                print("[心跳] 发送失败，连接已断开")
                break


def login_to_ipc(sock: socket.socket, username: str, password_raw: str) -> str | None:
    """执行光电 IPC 的登录流程"""
    # 1. 发送获取 Salt 请求
    salt_request = {
        "cmd": "userSaltGet",
        "param": {
            "username": username
        }
    }
    sock.sendall(pack_message(msg_id=1001, json_data=salt_request))
    print(f"[登录] 已发送 Salt 获取请求...")

    response = unpack_message(sock)
    if not response:
        print("[登录] 未收到 Salt 响应，连接断开")
        return None

    _, salt_data = response
    param = salt_data.get("param", {})
    salt = param.get("salt", "").strip()
    login_enc = param.get("loginEnc")  # 0: MD5, 1: HMAC-SHA256

    print(f"[登录] 获取 Salt 成功: salt='{salt}', 加密类型={login_enc}")

    # 2. 根据算法计算密码哈希值
    password_hash = ""
    if login_enc == 0:
        mix_str = password_raw + salt
        password_hash = hashlib.md5(mix_str.encode('utf-8')).hexdigest()
        print("[登录] 采用 MD5 加密")
    elif login_enc == 1:
        key = salt.encode('utf-8')
        msg = password_raw.encode('utf-8')
        password_hash = hmac.new(key, msg, hashlib.sha256).hexdigest()
        print("[登录] 采用 HMAC-SHA256 加密")
    else:
        print(f"[登录] 未知的加密类型: {login_enc}")
        return None

    # 3. 发送用户登录请求
    login_request = {
        "cmd": "userLogin",
        "param": {
            "username": username,
            "password": password_hash
        }
    }
    sock.sendall(pack_message(msg_id=1002, json_data=login_request))
    print("[登录] 已发送用户登录验证请求...")

    response = unpack_message(sock)
    if not response:
        print("[登录] 未收到登录响应")
        return None

    _, login_result = response
    login_param = login_result.get("param", {})
    res_ack = login_param.get("ackvalue")

    if res_ack == 100:
        token = login_param.get("token")
        print(f"[登录] 成功！获取到 Token: {token}")
        return token
    else:
        count = login_param.get("count", 0)
        lock_time = login_param.get("time", 0)
        print(f"[登录] 失败！错误码: {res_ack}, 剩余尝试次数: {count}, 锁定时间: {lock_time}秒")
        return None


import math

def calculate_target_gps(ptz_lat, ptz_lon, ptz_alt, pan, tilt, distance):
    """
    根据光电自身GPS、当前姿态角和距离，反推目标WGS84经纬度坐标
    :param ptz_lat: 光电自身纬度 (度, 例如 31.2304)
    :param ptz_lon: 光电自身经度 (度, 例如 121.4737)
    :param ptz_alt: 光电自身海拔高度 (米)
    :param pan:     绝对水平角 (度, 0为正北, 顺时针)
    :param tilt:    绝对俯仰角 (度, 水平为0, 仰角为正, 俯角为负)
    :param distance:目标斜距 (米)
    :return: (目标经度, 目标纬度, 目标高度)
    """
    if distance <= 0:
        return ptz_lon, ptz_lat, ptz_alt

    # 1. 弧度转换
    pan_rad = math.radians(pan)
    tilt_rad = math.radians(tilt)

    # 2. 计算相对光电的水平投影距离和垂直高度差
    horizontal_dist = distance * math.cos(tilt_rad)  # 地面投影距离
    delta_alt = distance * math.sin(tilt_rad)  # 高度差

    # 3. 在东北天(ENU)坐标系下的米级位移
    east = horizontal_dist * math.sin(pan_rad)  # 偏东距离
    north = horizontal_dist * math.cos(pan_rad)  # 偏北距离

    # 4. WGS84地球椭球体精确参数（消除地球曲率导致的经纬度偏离）
    a = 6378137.0  # 赤道半径
    b = 6356752.3142  # 极半径
    e2 = 1 - (b ** 2 / a ** 2)  # 第一偏心率平方

    lat_rad = math.radians(ptz_lat)

    # 计算当前纬度下的子午圈和卯酉圈曲率半径
    num = 1 - e2 * (math.sin(lat_rad) ** 2)
    radius_v = a / math.sqrt(num)  # 卯酉圈
    radius_m = a * (1 - e2) / (num ** 1.5)  # 子午圈

    # 5. 换算为经纬度变化量
    delta_lat = (north / radius_m) * (180.0 / math.pi)
    delta_lon = (east / (radius_v * math.cos(lat_rad))) * (180.0 / math.pi)

    # 6. 计算最终目标坐标
    target_lat = ptz_lat + delta_lat
    target_lon = ptz_lon + delta_lon
    target_alt = ptz_alt + delta_alt

    return round(target_lon, 6), round(target_lat, 6), round(target_alt, 1)

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 实际测试时请修改为光电外设的正确 IP
    sock.connect(('10.10.61.64', 39020))

    # 【修复 Bug 2】：传入文档推测的默认密码 'Abc.12345'
    token = login_to_ipc(sock, 'admin', 'Abc.12345')

    if not token:
        print("登录失败")
        sock.close()
        return

    token_dict = {"token": token}
    t = threading.Thread(target=heartbeat_thread, args=(sock, token_dict), daemon=True)
    t.start()

    print("\n[系统] 开始实时监听光电回传状态...")
    while True:
        # 【修复 Bug 3】：删除了 time.sleep(1)，交由 unpack_message 内部阻塞控制
        res = unpack_message(sock)
        if not res:
            print("[系统] 与光电断开连接")
            break

        msg_id, json_data = res
        cmd = json_data.get("cmd")
        param = json_data.get("param", {})

        # 4. 在这里动态捕获你想要的数据并打印出来
        if cmd == "PTZStatusReport":
            print(
                f"[云台数据] 水平角: {param.get('pan')}, 俯仰角: {param.get('tilt')}, 当前视场(倍率): {param.get('camView')}")
        elif cmd == "trackingReport":
            status_map = {0: "未跟踪", 1: "跟踪正常 ✅", 3: "目标失锁 ⚠️", 4: "目标丢失 ❌"}
            status_code = param.get("status", 0)
            print(
                f"[跟踪数据] 状态: {status_map.get(status_code, '未知')}, 目标距离: {param.get('trackDistance')}m, 脱靶量H: {param.get('offsetH')}")
        elif cmd == "ivpReport":
            print(f"[AI识别数据] 画面中检测到目标数量: {len(param.get('targets', []))}")


if __name__ == "__main__":
    main()