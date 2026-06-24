#!/usr/bin/env python3
"""
tcp_client.py — TCP 命令通道客户端

用于向机载电脑的 TCP 命令服务器发送指令：
    - STOP   : 触发紧急降落
    - PING   : 测试连接与响应
    - STATUS : 查询当前状态

使用方式:
    python3 tcp_client.py <server_ip> [port] [command]

示例:
    python3 tcp_client.py 192.168.1.100 9999 STOP
    python3 tcp_client.py 192.168.1.100 9999 PING
    python3 tcp_client.py 192.168.1.100 9999 STATUS
"""

import socket
import sys
import time

def send_command(server_ip, port, command, timeout=2.0):
    """
    向 TCP 命令服务器发送指令并接收响应

    Args:
        server_ip:  机载电脑 IP 地址
        port:       TCP 端口（默认 9999）
        command:    指令字符串（STOP / PING / STATUS）
        timeout:    超时时间（秒）

    Returns:
        (success: bool, response: str)
    """
    try:
        # 创建 TCP 连接
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((server_ip, port))
        print(f"✅ 已连接到 {server_ip}:{port}")

        # 发送指令（必须带换行符，与服务器协议一致）
        sock.send(f"{command}\n".encode())
        print(f"📨 已发送: {command}")

        # 接收响应
        response = sock.recv(1024).decode().strip()
        print(f"📨 收到响应: {response}")

        sock.close()
        return True, response

    except socket.timeout:
        print(f"❌ 连接超时 (>{timeout}s)")
        return False, "TIMEOUT"
    except ConnectionRefusedError:
        print(f"❌ 连接被拒绝: {server_ip}:{port} (服务未启动或端口错误)")
        return False, "CONNECTION_REFUSED"
    except Exception as e:
        print(f"❌ 异常: {e}")
        return False, str(e)


def interactive_mode(server_ip, port):
    """交互式模式：持续输入指令，直到输入 exit 退出"""
    print(f"\n🔌 进入交互模式 (连接 {server_ip}:{port})")
    print("可用指令: STOP, PING, STATUS, exit\n")

    # 先测试连接
    success, _ = send_command(server_ip, port, "PING", timeout=1.0)
    if not success:
        print("⚠️ 连接测试失败，请检查 IP 和端口")

    while True:
        try:
            cmd = input("> ").strip().upper()
            if cmd == "EXIT" or cmd == "QUIT":
                print("👋 退出")
                break
            if cmd in ("STOP", "PING", "STATUS"):
                send_command(server_ip, port, cmd)
            elif cmd == "":
                continue
            else:
                print(f"❌ 未知指令: {cmd}")
        except KeyboardInterrupt:
            print("\n👋 退出")
            break
        except EOFError:
            break


def main():
    # 解析命令行参数
    if len(sys.argv) < 2:
        print(__doc__)
        print("\n用法:")
        print("  python3 tcp_client.py <server_ip> [port] [command]")
        print("  python3 tcp_client.py <server_ip> [port]           # 进入交互模式")
        print("\n示例:")
        print("  python3 tcp_client.py 192.168.1.100 9999 STOP")
        print("  python3 tcp_client.py 192.168.1.100 9999          # 交互式")
        sys.exit(1)

    server_ip = "192.168.31.180"
    port = int(sys.argv[1]) if len(sys.argv) > 2 else 9999
    command = sys.argv[2].upper() if len(sys.argv) > 3 else None

    if command:
        # 一次性模式
        send_command(server_ip, port, command)
    else:
        # 交互式模式
        interactive_mode(server_ip, port)


if __name__ == "__main__":
    main()