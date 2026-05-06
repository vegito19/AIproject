"""
Suspicion Engine — Trust/Suspicion Scoring
============================================
Calculates dynamic suspicion levels for every player
based on observable events from the memory system.

Every player starts at 50% (neutral). Suspicion changes
based on behavioral evidence with safeguards against
false accusations.
"""

import time


class SuspicionEngine:
    """Evidence-based suspicion scoring system."""

    # Starting suspicion for all players
    DEFAULT_SUSPICION = 50.0
    # Max and min bounds
    MAX_SUSPICION = 100.0
    MIN_SUSPICION = 0.0
    # Suspicion decay per interval (reduces over time)
    DECAY_AMOUNT = 2.0
    DECAY_INTERVAL = 30.0  # seconds

    # --- Suspicion change rules ---
    SUSPICION_INCREASES = {
        'proximity_to_body':     20,
        'running_from_body':     35,
        'fake_task':             25,
        'idle_too_long':         10,
        'near_vent':             15,
        'direction_change':      10,
        'alone_with_victim':     40,
        'contradicting_claim':   30,
    }

    SUSPICION_DECREASES = {
        'task_completed':        -15,
        'group_presence':        -10,
        'quick_body_report':     -10,
        'correct_accusation':    -20,
    }

    # Confidence levels for decision thresholds
    CONFIDENCE_LEVELS = {
        'low': 0.3,
        'medium': 0.5,
        'high': 0.7,
        'critical': 0.9,
    }

    def __init__(self, memory_system):
        self.memory = memory_system
        self.suspicion_scores = {}   # player_id -> float
        self.last_decay_time = time.time()
        self.processed_events = set()  # event ids to avoid double-counting

    def register_player(self, player_id):
        """Initialize suspicion for a new player."""
        if player_id not in self.suspicion_scores:
            self.suspicion_scores[player_id] = self.DEFAULT_SUSPICION

    def get_suspicion(self, player_id):
        """Get current suspicion level for a player."""
        self.register_player(player_id)
        return self.suspicion_scores[player_id]

    def get_all_suspicions(self):
        """Get suspicion levels for all tracked players."""
        return dict(self.suspicion_scores)

    def get_most_suspicious(self):
        """Get the player with the highest suspicion."""
        if not self.suspicion_scores:
            return None, 0
        pid = max(self.suspicion_scores, key=self.suspicion_scores.get)
        return pid, self.suspicion_scores[pid]

    def update(self):
        """
        Main update loop. Processes new events from memory and
        updates suspicion scores accordingly.
        """
        self._process_new_events()
        self._apply_decay()
        # Log current suspicion levels to memory
        for pid, score in self.suspicion_scores.items():
            self.memory.log_suspicion_change(pid, score)

    def _process_new_events(self):
        """Process unprocessed events from memory."""
        for event in self.memory.events:
            event_key = id(event)
            if event_key in self.processed_events:
                continue
            self.processed_events.add(event_key)

            pid = event.player_id
            self.register_player(pid)
            etype = event.event_type

            # Apply suspicion increases
            if etype in self.SUSPICION_INCREASES:
                change = self.SUSPICION_INCREASES[etype]
                # Weight by event confidence and age
                weighted = change * event.confidence * event.age_weight
                self._modify_suspicion(pid, weighted)

            # Apply suspicion decreases
            if etype in self.SUSPICION_DECREASES:
                change = self.SUSPICION_DECREASES[etype]
                weighted = change * event.confidence * event.age_weight
                self._modify_suspicion(pid, weighted)

    def _modify_suspicion(self, player_id, amount):
        """Change a player's suspicion score with clamping."""
        self.register_player(player_id)
        self.suspicion_scores[player_id] += amount
        self.suspicion_scores[player_id] = max(
            self.MIN_SUSPICION,
            min(self.MAX_SUSPICION, self.suspicion_scores[player_id])
        )

    def _apply_decay(self):
        """Reduce suspicion over time for all players (prevents runaway scores)."""
        now = time.time()
        if now - self.last_decay_time < self.DECAY_INTERVAL:
            return
        self.last_decay_time = now
        for pid in self.suspicion_scores:
            if self.suspicion_scores[pid] > self.DEFAULT_SUSPICION:
                self.suspicion_scores[pid] = max(
                    self.DEFAULT_SUSPICION,
                    self.suspicion_scores[pid] - self.DECAY_AMOUNT
                )

    def get_confidence_level(self, player_id):
        """
        Determine confidence level for a suspicion score.

        Returns:
            str: 'low', 'medium', 'high', or 'critical'
        """
        evidence_count = self.memory.count_evidence_against(player_id)
        suspicion = self.get_suspicion(player_id)

        if evidence_count >= 4 and suspicion >= 85:
            return 'critical'
        elif evidence_count >= 3 and suspicion >= 70:
            return 'high'
        elif evidence_count >= 2 and suspicion >= 55:
            return 'medium'
        else:
            return 'low'

    def should_accuse(self, player_id):
        """
        Determine if the AI has enough evidence to accuse a player.
        Requires both high suspicion AND sufficient evidence count.

        Returns:
            bool: True if accusation is warranted
        """
        suspicion = self.get_suspicion(player_id)
        confidence = self.get_confidence_level(player_id)
        evidence_count = self.memory.count_evidence_against(player_id)

        return (
            suspicion >= 75 and
            confidence in ('high', 'critical') and
            evidence_count >= 2
        )

    def reset_player(self, player_id):
        """Reset suspicion for an ejected or cleared player."""
        if player_id in self.suspicion_scores:
            self.suspicion_scores[player_id] = self.DEFAULT_SUSPICION

    def reset_all(self):
        """Reset all suspicion scores."""
        for pid in self.suspicion_scores:
            self.suspicion_scores[pid] = self.DEFAULT_SUSPICION
        self.processed_events.clear()
