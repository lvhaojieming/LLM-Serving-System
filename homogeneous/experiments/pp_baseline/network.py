"""A narrow route for the peer's primary Gloo address on a multi-address NIC."""
import errno
import socket
import struct


def attribute(kind, data):
    size = 4+len(data)
    return struct.pack('HH', size, kind)+data+b'\0'*((-size)%4)


def ensure_peer_routes(node, peers):
    retained = []
    for route in node.get('peer_routes', []):
        # Only a declared peer's own secondary/primary pair is authorized here.
        if not any(peer['host'] == route['gateway'] and peer.get('interface_primary_ip') == route['destination'] for peer in peers):
            raise ValueError('Route must target a configured peer interface address')
        index = socket.if_nametoindex(node['interface'])
        destination = attribute(1, socket.inet_aton(route['destination']))
        payload = struct.pack('BBBBBBBBI', socket.AF_INET,32,0,0,254,4,0,1,0)+destination
        payload += attribute(5, socket.inet_aton(route['gateway']))+attribute(4, struct.pack('I', index))
        request = struct.pack('IHHII',16+len(payload),24,1|4|0x400|0x200,1,0)+payload
        with socket.socket(socket.AF_NETLINK,socket.SOCK_RAW,socket.NETLINK_ROUTE) as client:
            client.bind((0,0)); client.send(request)
            reply = client.recv(4096)
            error = struct.unpack('i',reply[16:20])[0]
            if error not in (0, -errno.EEXIST):
                raise OSError(-error, 'Cannot install experiment peer route')
        # Never replace an existing route; verify that its next hop is the peer.
        query = struct.pack('BBBBBBBBI', socket.AF_INET,32,0,0,254,0,0,0,0)+destination
        with socket.socket(socket.AF_NETLINK,socket.SOCK_RAW,socket.NETLINK_ROUTE) as client:
            client.bind((0,0)); client.send(struct.pack('IHHII',16+len(query),26,1,2,0)+query)
            reply = client.recv(4096)
        if struct.unpack('H',reply[4:6])[0] != 24:
            raise RuntimeError('Peer route lookup failed')
        values = {}; cursor = 28
        while cursor+4 <= len(reply):
            size, kind = struct.unpack('HH',reply[cursor:cursor+4])
            if size < 4 or cursor+size > len(reply):
                raise RuntimeError('Invalid route response')
            values[kind] = reply[cursor+4:cursor+size]; cursor += (size+3)&~3
        if values.get(5) != socket.inet_aton(route['gateway']) or values.get(4) != struct.pack('I', index):
            raise RuntimeError('Existing route conflicts; it was preserved')
        retained.append(dict(route, prefix=32, interface=node['interface'], created=error==0))
    return retained
