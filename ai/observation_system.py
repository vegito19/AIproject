"""
Observation System — Player Behavior Tracking
===============================================
Watches all visible players and logs behavioral events
to the memory system. This is the AI's "perception layer".

The AI only observes what's within its vision radius (no cheating).
"""

import math
import time


class ObservationSystem:
    """Monitors player behavior using only observable events."""

    # AI vision radius (same as player light radius)
    VISION_RADIUS = 500
    # How close to a body counts as "near body"
    BODY_PROXIMITY_RADIUS = 150
    # How close to a vent counts as "near vent"
    VENT_PROXIMITY_RADIUS = 100
    # Time (seconds) of no movement to be considered "idle"
    IDLE_THRESHOLD = 8.0
    # How often to check for observations (seconds)
    OBSERVATION_INTERVAL = 0.5

    # Vent locations from game.py
    VENT_LOCATIONS = [
        (3898, 791), (5309, 1144), (5309, 1525), (4513, 1525),
        (4531, 2459), (3694, 1942), (2220, 1711), (1580, 2407),
        (1887, 1578), (931, 1626), (802, 1151), (1586, 460),
        (2121, 1249), (4447, 363)
    ]

    def __init__(self, memory_system):
        self.memory = memory_system
        self.last_observation_time = 0
        self.player_idle_timers = {}   # player_id -> last_move_time
        self.known_bodies = set()      # set of (x, y) corpse positions

    def observe(self, ai_x, ai_y, players_data, bodies_data=None):
        """
        Main observation loop. Called every frame by the AI agent.

        Args:
            ai_x, ai_y: AI's current position
            players_data: list of dicts with keys:
                          {id, x, y, alive, colour, tasks_completed}
            bodies_data: list of (x, y) corpse positions (optional)
        """
        now = time.time()
        if now - self.last_observation_time < self.OBSERVATION_INTERVAL:
            return
        self.last_observation_time = now

        # Register and update all visible players
        for p in players_data:
            pid = p['id']
            px, py = p['x'], p['y']

            # Check if player is within vision
            dist = math.sqrt((ai_x - px)**2 + (ai_y - py)**2)
            if dist > self.VISION_RADIUS:
                continue

            self.memory.update_player_position(pid, px, py)
            profile = self.memory.get_player_profile(pid)
            profile.alive = p.get('alive', True)

            # --- Check for suspicious behaviors ---
            self._check_near_body(pid, px, py, bodies_data)
            self._check_near_vent(pid, px, py)
            self._check_idle(pid, px, py, now)
            self._check_direction_changes(pid, profile)

    def _check_near_body(self, pid, px, py, bodies_data):
        """Check if a player is near a dead body."""
        if not bodies_data:
            return
        for bx, by in bodies_data:
            dist = math.sqrt((px - bx)**2 + (py - by)**2)
            if dist <= self.BODY_PROXIMITY_RADIUS:
                self.memory.log_event(
                    'proximity_to_body', pid, (px, py),
                    details={'body_pos': (bx, by), 'distance': dist},
                    confidence=0.8
                )

    def _check_near_vent(self, pid, px, py):
        """Check if player is suspiciously close to a vent."""
        for vx, vy in self.VENT_LOCATIONS:
            dist = math.sqrt((px - vx)**2 + (py - vy)**2)
            if dist <= self.VENT_PROXIMITY_RADIUS:
                profile = self.memory.get_player_profile(pid)
                profile.times_seen_near_vent += 1
                self.memory.log_event(
                    'near_vent', pid, (px, py),
                    details={'vent_pos': (vx, vy)},
                    confidence=0.6
                )
                break

    def _check_idle(self, pid, px, py, now):
        """Check if player has been idle too long."""
        if pid not in self.player_idle_timers:
            self.player_idle_timers[pid] = {'time': now, 'pos': (px, py)}
            return

        last = self.player_idle_timers[pid]
        dist = math.sqrt((px - last['pos'][0])**2 + (py - last['pos'][1])**2)

        if dist < 10:  # barely moved
            idle_duration = now - last['time']
            if idle_duration > self.IDLE_THRESHOLD:
                self.memory.log_event(
                    'idle_too_long', pid, (px, py),
                    details={'idle_seconds': idle_duration},
                    confidence=0.4
                )
                # Reset timer so we don't spam
                self.player_idle_timers[pid]['time'] = now
        else:
            self.player_idle_timers[pid] = {'time': now, 'pos': (px, py)}

    def _check_direction_changes(self, pid, profile):
        """Detect erratic movement (frequently changing direction)."""
        history = profile.positions_history
        if len(history) < 4:
            return

        # Use the last 4 known positions and safely compute direction deltas
        last_positions = history[-4:]
        changes = 0
        for i in range(len(last_positions) - 2):
            p0 = last_positions[i]
            p1 = last_positions[i+1]
            p2 = last_positions[i+2]
            d1 = (p1[0] - p0[0], p1[1] - p0[1])
            d2 = (p2[0] - p1[0], p2[1] - p1[1])
            # Dot product — negative means direction reversal
            dot = d1[0] * d2[0] + d1[1] * d2[1]
            if dot < 0:
                changes += 1

        if changes >= 2:
            profile.direction_changes += 1
            if profile.direction_changes % 5 == 0:  # Log every 5th instance
                self.memory.log_event(
                    'direction_change', pid,
                    profile.last_known_position or (0, 0),
                    details={'total_changes': profile.direction_changes},
                    confidence=0.3
                )

    def detect_body(self, ai_x, ai_y, bodies_data):
        """
        Check if the AI can see a new dead body.

        Returns:
            tuple or None: (x, y) of newly discovered body
        """
        if not bodies_data:
            return None
        for bx, by in bodies_data:
            body_key = (int(bx), int(by))
            if body_key in self.known_bodies:
                continue
            dist = math.sqrt((ai_x - bx)**2 + (ai_y - by)**2)
            if dist <= self.VISION_RADIUS:
                self.known_bodies.add(body_key)
                return (bx, by)
        return None

    def check_alone_with_victim(self, suspect_id, victim_pos, all_players, radius=300):
        """
        Check if a player was alone with the victim before death.

        Returns:
            bool: True if suspect was the only one near the victim
        """
        nearby = []
        for p in all_players:
            if not p.get('alive', True):
                continue
            dist = math.sqrt(
                (p['x'] - victim_pos[0])**2 + (p['y'] - victim_pos[1])**2
            )
            if dist <= radius:
                nearby.append(p['id'])

        # Suspect was alone with victim if they're the only one nearby
        return len(nearby) == 1 and suspect_id in nearby
