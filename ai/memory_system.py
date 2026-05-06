"""
Memory System — Event Logging & Behavioral Memory
===================================================
Stores all observations, events, and behavioral profiles
so the AI can reason about past events during meetings.

Memory Types:
  - Short-term:      Current round real-time decisions
  - Event log:       Full game evidence trail
  - Suspicion history: Track suspicion changes over time
  - Player profiles: Behavioral patterns per player
"""

import time


class MemoryEvent:
    """A single recorded event in the AI's memory."""

    def __init__(self, event_type, player_id, location, details=None, confidence=1.0):
        """
        Args:
            event_type (str): One of 'position', 'kill', 'body_found', 'task',
                              'meeting', 'vote', 'proximity', 'isolation', 'vent',
                              'idle', 'direction_change', 'group'
            player_id (int or str): Identifier of the player involved
            location (tuple): (x, y) coordinates where event occurred
            details (dict): Additional event-specific data
            confidence (float): 0.0 to 1.0 — how confident the AI is about this event
        """
        self.timestamp = time.time()
        self.event_type = event_type
        self.player_id = player_id
        self.location = location
        self.details = details or {}
        self.confidence = confidence
        self.age_weight = 1.0  # decays over time

    def __repr__(self):
        return (f"MemoryEvent(type={self.event_type}, player={self.player_id}, "
                f"loc={self.location}, conf={self.confidence:.2f}, "
                f"age_wt={self.age_weight:.2f})")


class PlayerProfile:
    """Behavioral profile for a single player."""

    def __init__(self, player_id):
        self.player_id = player_id
        self.last_known_position = None
        self.last_seen_time = None
        self.positions_history = []         # list of (x, y, timestamp)
        self.tasks_completed_count = 0
        self.times_seen_near_body = 0
        self.times_seen_near_vent = 0
        self.times_seen_alone_with_victim = 0
        self.times_reported_body = 0
        self.times_called_meeting = 0
        self.vote_history = []              # list of (who_they_voted_for, round)
        self.idle_time_total = 0.0          # seconds spent idle
        self.direction_changes = 0          # erratic movement counter
        self.rooms_visited = []             # ordered list of room names
        self.alive = True
        self.is_ejected = False

    def update_position(self, x, y, timestamp):
        """Record a new position observation."""
        self.last_known_position = (x, y)
        self.last_seen_time = timestamp
        self.positions_history.append((x, y, timestamp))
        # Keep only last 100 positions to prevent memory bloat
        if len(self.positions_history) > 100:
            self.positions_history.pop(0)

    def get_movement_speed(self):
        """Calculate approximate movement speed from recent positions."""
        if len(self.positions_history) < 2:
            return 0.0
        p1 = self.positions_history[-2]
        p2 = self.positions_history[-1]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dt = p2[2] - p1[2]
        if dt <= 0:
            return 0.0
        distance = (dx ** 2 + dy ** 2) ** 0.5
        return distance / dt

    def get_movement_direction(self):
        """Get the current movement direction vector."""
        if len(self.positions_history) < 2:
            return (0, 0)
        p1 = self.positions_history[-2]
        p2 = self.positions_history[-1]
        return (p2[0] - p1[0], p2[1] - p1[1])


class MemorySystem:
    """
    Central memory storage for the AI agent.
    Maintains event logs, player profiles, and suspicion history.
    """

    # Maximum events to store (prevents unbounded memory growth)
    MAX_EVENTS = 500
    # How fast old events lose relevance (weight multiplier per 30 seconds)
    AGE_DECAY_RATE = 0.95
    # Time interval for age decay (seconds)
    AGE_DECAY_INTERVAL = 30.0

    def __init__(self):
        self.events = []                     # list of MemoryEvent
        self.player_profiles = {}            # player_id -> PlayerProfile
        self.suspicion_history = {}          # player_id -> list of (timestamp, suspicion)
        self.round_number = 0
        self.game_start_time = time.time()
        self.last_decay_time = time.time()
        self.meeting_log = []                # list of meeting summaries

    def reset(self):
        """Reset all memory for a new game."""
        self.events.clear()
        self.player_profiles.clear()
        self.suspicion_history.clear()
        self.round_number = 0
        self.game_start_time = time.time()
        self.last_decay_time = time.time()
        self.meeting_log.clear()

    def register_player(self, player_id):
        """Create a profile for a newly discovered player."""
        if player_id not in self.player_profiles:
            self.player_profiles[player_id] = PlayerProfile(player_id)
        if player_id not in self.suspicion_history:
            self.suspicion_history[player_id] = []

    def log_event(self, event_type, player_id, location, details=None, confidence=1.0):
        """
        Record an event in memory.

        Args:
            event_type (str): Type of event observed
            player_id: Player involved
            location (tuple): (x, y) position
            details (dict): Extra data
            confidence (float): How sure the AI is (0.0 - 1.0)

        Returns:
            MemoryEvent: The created event
        """
        event = MemoryEvent(event_type, player_id, location, details, confidence)
        self.events.append(event)

        # Enforce max events limit
        if len(self.events) > self.MAX_EVENTS:
            self.events.pop(0)

        # Ensure player has a profile
        self.register_player(player_id)

        return event

    def update_player_position(self, player_id, x, y):
        """Update a player's position in their profile."""
        self.register_player(player_id)
        self.player_profiles[player_id].update_position(x, y, time.time())

    def get_player_profile(self, player_id):
        """Get the behavioral profile for a player."""
        self.register_player(player_id)
        return self.player_profiles[player_id]

    def get_events_by_type(self, event_type, player_id=None):
        """Retrieve all events of a given type, optionally filtered by player."""
        results = [e for e in self.events if e.event_type == event_type]
        if player_id is not None:
            results = [e for e in results if e.player_id == player_id]
        return results

    def get_events_since(self, timestamp):
        """Get all events that occurred after a given timestamp."""
        return [e for e in self.events if e.timestamp >= timestamp]

    def get_recent_events(self, count=10):
        """Get the N most recent events."""
        return self.events[-count:]

    def log_suspicion_change(self, player_id, suspicion_value):
        """Record a suspicion change for tracking over time."""
        self.register_player(player_id)
        self.suspicion_history[player_id].append((time.time(), suspicion_value))

    def log_meeting(self, meeting_summary):
        """
        Record a meeting summary.

        Args:
            meeting_summary (dict): {
                'round': int,
                'caller': player_id,
                'reason': str,
                'votes': {player_id: voted_for},
                'ejected': player_id or None,
                'timestamp': float
            }
        """
        self.meeting_log.append(meeting_summary)
        self.round_number += 1

    def apply_age_decay(self):
        """
        Reduce the weight of older events over time.
        Called periodically by the AI agent.
        """
        now = time.time()
        if now - self.last_decay_time < self.AGE_DECAY_INTERVAL:
            return

        self.last_decay_time = now
        for event in self.events:
            age_seconds = now - event.timestamp
            decay_periods = age_seconds / self.AGE_DECAY_INTERVAL
            event.age_weight = self.AGE_DECAY_RATE ** decay_periods

    def get_evidence_for_player(self, player_id):
        """
        Collect all evidence (suspicious events) related to a specific player.
        Returns events sorted by relevance (weighted by confidence × age_weight).

        Returns:
            list[MemoryEvent]: Sorted evidence list
        """
        suspicious_types = {
            'proximity_to_body', 'running_from_body', 'fake_task',
            'idle_too_long', 'near_vent', 'direction_change',
            'alone_with_victim', 'contradicting_claim'
        }
        evidence = [
            e for e in self.events
            if e.player_id == player_id and e.event_type in suspicious_types
        ]
        # Sort by relevance (highest first)
        evidence.sort(key=lambda e: e.confidence * e.age_weight, reverse=True)
        return evidence

    def get_alibi_for_self(self, ai_player_id):
        """
        Collect evidence of what the AI was doing (for self-defense in meetings).

        Returns:
            list[MemoryEvent]: AI's own activity log
        """
        return [
            e for e in self.events
            if e.player_id == ai_player_id and e.event_type in {'task_completed', 'position', 'group'}
        ]

    def count_evidence_against(self, player_id):
        """Count the number of suspicious events for a player."""
        return len(self.get_evidence_for_player(player_id))

    def get_players_in_room(self, room_center, radius, exclude_id=None):
        """
        Get all players whose last known position is within a room's radius.

        Args:
            room_center (tuple): (x, y) center of the room
            radius (float): Detection radius
            exclude_id: Player to exclude (typically self)

        Returns:
            list[int]: Player IDs in the room
        """
        players_in_room = []
        for pid, profile in self.player_profiles.items():
            if pid == exclude_id:
                continue
            if profile.last_known_position is None:
                continue
            dx = profile.last_known_position[0] - room_center[0]
            dy = profile.last_known_position[1] - room_center[1]
            dist = (dx ** 2 + dy ** 2) ** 0.5
            if dist <= radius:
                players_in_room.append(pid)
        return players_in_room
