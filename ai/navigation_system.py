"""
Navigation System — Waypoint Pathfinding & Movement
=====================================================
Handles AI movement using a waypoint graph derived from the game map.
"""

import math
import random
import time


class Waypoint:
    """A navigation point on the map."""
    def __init__(self, name, x, y, room=None):
        self.name = name
        self.x = x
        self.y = y
        self.room = room
        self.neighbors = []

    def position(self):
        return (self.x, self.y)

    def distance_to(self, ox, oy):
        return math.sqrt((self.x - ox)**2 + (self.y - oy)**2)


class NavigationSystem:
    """Waypoint-based pathfinding for the AI agent."""

    AI_SPEED = 350
    WAYPOINT_REACH_THRESHOLD = 40
    STUCK_TIMEOUT = 3.0

    def __init__(self):
        self.waypoints = {}
        self.current_path = []
        self.current_waypoint_index = 0
        self.target_position = None
        self.is_moving = False
        self.last_position = None
        self.stuck_timer = 0.0
        self.last_update_time = time.time()
        self._build_waypoint_graph()

    def _build_waypoint_graph(self):
        """Define waypoints and connections from the game map."""
        rooms = [
            ('cafeteria', 3277, 658), ('medbay', 2338, 1147),
            ('security', 1806, 1279), ('reactor', 880, 1474),
            ('upper_engine', 1360, 699), ('lower_engine', 1360, 2180),
            ('electrical', 2425, 1950), ('storage', 3175, 2308),
            ('admin', 3920, 1775), ('communications', 3865, 2650),
            ('oxygen', 4190, 1220), ('cockpit', 5405, 1340),
            ('weapons', 4500, 600),
        ]
        corridors = [
            ('cafe_south', 3277, 1000), ('cafe_east', 3900, 658),
            ('medbay_north', 2338, 900), ('medbay_south', 2338, 1400),
            ('security_west', 1500, 1279), ('upper_eng_south', 1360, 1000),
            ('reactor_north', 880, 1100), ('reactor_south', 880, 1800),
            ('lower_eng_north', 1360, 1900), ('electrical_north', 2425, 1600),
            ('electrical_west', 2000, 1950), ('storage_north', 3175, 1900),
            ('storage_east', 3600, 2308), ('admin_north', 3920, 1400),
            ('admin_south', 3920, 2100), ('oxygen_west', 3900, 1220),
            ('weapons_south', 4500, 900), ('cockpit_west', 5000, 1340),
            ('central_hallway', 2800, 1400),
        ]

        for name, x, y in rooms:
            self.waypoints[name] = Waypoint(name, x, y, room=name)
        for name, x, y in corridors:
            self.waypoints[name] = Waypoint(name, x, y, room=None)

        connections = [
            ('cafeteria', 'cafe_south'), ('cafeteria', 'cafe_east'),
            ('cafeteria', 'medbay_north'), ('cafe_east', 'weapons_south'),
            ('cafe_east', 'oxygen_west'), ('medbay_north', 'medbay'),
            ('medbay', 'medbay_south'), ('medbay_north', 'upper_eng_south'),
            ('medbay_south', 'central_hallway'), ('cafe_south', 'central_hallway'),
            ('central_hallway', 'electrical_north'), ('central_hallway', 'storage_north'),
            ('central_hallway', 'admin_north'), ('central_hallway', 'medbay_south'),
            ('central_hallway', 'security_west'), ('security_west', 'security'),
            ('security', 'upper_eng_south'), ('security', 'reactor_north'),
            ('upper_engine', 'upper_eng_south'), ('upper_eng_south', 'reactor_north'),
            ('upper_eng_south', 'security_west'), ('reactor_north', 'reactor'),
            ('reactor', 'reactor_south'), ('reactor_south', 'lower_eng_north'),
            ('lower_eng_north', 'lower_engine'), ('lower_eng_north', 'electrical_west'),
            ('electrical_north', 'electrical'), ('electrical', 'electrical_west'),
            ('electrical_north', 'storage_north'), ('storage_north', 'storage'),
            ('storage', 'storage_east'), ('storage_east', 'admin_south'),
            ('storage_east', 'communications'), ('admin_north', 'admin'),
            ('admin', 'admin_south'), ('admin_south', 'communications'),
            ('oxygen_west', 'oxygen'), ('oxygen', 'admin_north'),
            ('weapons_south', 'weapons'), ('weapons', 'cockpit_west'),
            ('cockpit_west', 'cockpit'), ('cockpit_west', 'oxygen'),
        ]
        for a, b in connections:
            if a in self.waypoints and b in self.waypoints:
                self.waypoints[a].neighbors.append(b)
                self.waypoints[b].neighbors.append(a)

    def find_nearest_waypoint(self, x, y):
        nearest, min_d = None, float('inf')
        for wp in self.waypoints.values():
            d = wp.distance_to(x, y)
            if d < min_d:
                min_d, nearest = d, wp
        return nearest

    def find_path(self, sx, sy, tx, ty):
        """BFS pathfinding on the waypoint graph."""
        s = self.find_nearest_waypoint(sx, sy)
        e = self.find_nearest_waypoint(tx, ty)
        if not s or not e:
            return []
        if s.name == e.name:
            return [e.name]
        visited = {s.name}
        queue = [(s.name, [s.name])]
        while queue:
            cur, path = queue.pop(0)
            for nb in self.waypoints[cur].neighbors:
                if nb not in visited:
                    new_path = path + [nb]
                    if nb == e.name:
                        return new_path
                    visited.add(nb)
                    queue.append((nb, new_path))
        return [e.name]

    def set_destination(self, cx, cy, tx, ty):
        self.current_path = self.find_path(cx, cy, tx, ty)
        self.current_waypoint_index = 0
        self.target_position = (tx, ty)
        self.is_moving = len(self.current_path) > 0
        self.stuck_timer = 0.0
        self.last_position = (cx, cy)

    def set_random_destination(self, cx, cy):
        room_wps = [wp for wp in self.waypoints.values() if wp.room]
        if room_wps:
            t = random.choice(room_wps)
            self.set_destination(cx, cy, t.x, t.y)

    def get_velocity(self, cx, cy, dt):
        if not self.is_moving or not self.current_path:
            return (0, 0)
        if self.current_waypoint_index >= len(self.current_path):
            self.is_moving = False
            return (0, 0)
        twp = self.waypoints[self.current_path[self.current_waypoint_index]]
        if twp.distance_to(cx, cy) < self.WAYPOINT_REACH_THRESHOLD:
            self.current_waypoint_index += 1
            if self.current_waypoint_index >= len(self.current_path):
                self.is_moving = False
                return (0, 0)
            twp = self.waypoints[self.current_path[self.current_waypoint_index]]
        dx, dy = twp.x - cx, twp.y - cy
        length = math.sqrt(dx**2 + dy**2)
        if length < 1:
            return (0, 0)
        vx = (dx / length) * self.AI_SPEED
        vy = (dy / length) * self.AI_SPEED
        # Stuck detection
        now = time.time()
        if self.last_position:
            md = math.sqrt((cx - self.last_position[0])**2 + (cy - self.last_position[1])**2)
            self.stuck_timer = self.stuck_timer + (now - self.last_update_time) if md < 5 else 0.0
        self.last_position = (cx, cy)
        self.last_update_time = now
        if self.stuck_timer > self.STUCK_TIMEOUT:
            self.stuck_timer = 0.0
            self.set_random_destination(cx, cy)
        return (vx, vy)

    def has_reached_destination(self):
        return not self.is_moving

    def get_current_room(self, x, y):
        radii = {'cafeteria': 400, 'medbay': 300, 'security': 250, 'reactor': 350,
                 'upper_engine': 300, 'lower_engine': 300, 'electrical': 350,
                 'storage': 400, 'admin': 300, 'communications': 250,
                 'oxygen': 200, 'cockpit': 250, 'weapons': 300}
        for wp in self.waypoints.values():
            if wp.room and wp.room in radii and wp.distance_to(x, y) <= radii[wp.room]:
                return wp.room
        return None

    def get_room_position(self, room_name):
        for wp in self.waypoints.values():
            if wp.room == room_name:
                return (wp.x, wp.y)
        return None
