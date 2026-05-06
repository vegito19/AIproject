import select
import socket

# Minimal compatibility shim for removed `asyncore` module.
# Implements dispatcher, dispatcher_with_send and loop used by server.py.

_map = {}

class dispatcher:
    def __init__(self, sock=None):
        self.socket = None
        if sock is not None:
            self.socket = sock
            _map[self.socket] = self

    def create_socket(self, family, type):
        self.socket = socket.socket(family, type)
        self.socket.setblocking(False)
        _map[self.socket] = self

    def bind(self, addr):
        self.socket.bind(addr)

    def listen(self, backlog):
        self.socket.listen(backlog)

    def accept(self):
        return self.socket.accept()

    def close(self):
        try:
            _map.pop(self.socket, None)
            self.socket.close()
        except Exception:
            pass

    # callback placeholders
    def handle_accept(self):
        pass

    def handle_read(self):
        pass

class dispatcher_with_send(dispatcher):
    def __init__(self, sock=None):
        super().__init__(sock)

    def recv(self, n):
        try:
            return self.socket.recv(n)
        except BlockingIOError:
            return b''

    def send(self, data):
        try:
            return self.socket.send(data)
        except Exception:
            return 0

def loop(timeout=1.0, use_poll=False):
    while True:
        if not _map:
            break
        rlist = []
        wlist = []
        xlist = []
        for sock, handler in list(_map.items()):
            rlist.append(sock)
            xlist.append(sock)

        try:
            ready_r, ready_w, ready_x = select.select(rlist, wlist, xlist, timeout)
        except Exception:
            break

        for sock in ready_r:
            handler = _map.get(sock)
            if handler is None:
                continue
            # If this is a listening socket, call handle_accept
            try:
                if sock.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN):
                    try:
                        handler.handle_accept()
                    except TypeError:
                        # handle_accept may expect different signature
                        handler.handle_accept()
                    continue
            except Exception:
                pass

            # Otherwise call handle_read on handler (if exists)
            try:
                handler.handle_read()
            except Exception:
                try:
                    handler.close()
                except Exception:
                    pass

        for sock in ready_x:
            handler = _map.get(sock)
            if handler:
                try:
                    handler.close()
                except Exception:
                    pass
