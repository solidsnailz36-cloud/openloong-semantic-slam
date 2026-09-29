#!/usr/bin/env python3
"""Distinguish "the box is off" from "the port moved" from "a firewall drops us".

The three failure modes look identical to a user ("连不上") but have completely
different fixes, so probe the TCP layer and read the errno:

  ConnectionRefusedError (WinError 10061) -> host is UP, nothing listening on
      that port.  The service is not running (or moved to another port).
  socket.timeout         (WinError 10060) -> no answer at all: the host is
      stopped, or the NAT mapping is gone, or something drops the packets.
  success                                 -> it is up.

Also probes the usual local-service ports, because on these metered GPU boxes
the SSH port and the Jupyter URL are re-assigned on every restart.
"""
import socket
import sys

HOST = sys.argv[1] if len(sys.argv) > 1 else "183.239.162.232"
PORTS = [int(p) for p in (sys.argv[2].split(",") if len(sys.argv) > 2
                          else ["30032", "22", "8888", "8889", "8000", "8080", "6006"])]
TIMEOUT = 8.0

print("target %s   timeout %.0f s\n" % (HOST, TIMEOUT))
for p in PORTS:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(TIMEOUT)
    try:
        s.connect((HOST, p))
        print("  %-6d OPEN      <- something is listening" % p)
    except ConnectionRefusedError as e:
        print("  %-6d REFUSED   <- host is UP, service not running  (%s)"
              % (p, e.errno))
    except socket.timeout:
        print("  %-6d TIMEOUT   <- no answer: stopped host / moved NAT mapping / filtered"
              % p)
    except OSError as e:
        print("  %-6d ERROR %s  (%s)" % (p, e.errno, e))
    finally:
        s.close()
