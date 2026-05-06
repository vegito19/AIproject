"""
Decision Engine — Meeting Triggers & Voting Logic
===================================================
Decides when to call emergency meetings, how to vote,
and manages the AI's priority system.
"""

import time


class DecisionEngine:
    """Handles meeting triggers, voting, and priority-based decisions."""

    # Meeting thresholds
    MEETING_SUSPICION_THRESHOLD = 90
    MEETING_EVIDENCE_MIN = 3
    MEETING_COOLDOWN = 60.0   # seconds between meetings

    # Voting thresholds
    VOTE_SUSPICION_THRESHOLD = 75
    SKIP_THRESHOLD = 60       # below this, skip vote

    # Priority weights (higher = more important)
    PRIORITIES = {
        'survive': 5,
        'complete_tasks': 4,
        'detect_impostor': 3,
        'protect_crew': 2,
        'avoid_false_accusations': 1,
    }

    def __init__(self, suspicion_engine, memory_system):
        self.suspicion = suspicion_engine
        self.memory = memory_system
        self.last_meeting_time = 0
        self.meetings_called = 0
        self.votes_cast = []

    def should_call_meeting(self, is_emergency_active=False):
        """
        Determine if the AI should call an emergency meeting.

        Conditions (ALL must be true):
        1. Suspicion of any player > 90%
        2. Confidence level >= 'high'
        3. At least 3 pieces of evidence
        4. Meeting cooldown is over
        5. No active emergency

        Returns:
            tuple: (should_call: bool, target_player_id, reason: str)
        """
        if is_emergency_active:
            return (False, None, "")

        now = time.time()
        if now - self.last_meeting_time < self.MEETING_COOLDOWN:
            return (False, None, "")

        # Find the most suspicious player
        pid, score = self.suspicion.get_most_suspicious()
        if pid is None:
            return (False, None, "")

        if score < self.MEETING_SUSPICION_THRESHOLD:
            return (False, None, "")

        confidence = self.suspicion.get_confidence_level(pid)
        if confidence not in ('high', 'critical'):
            return (False, None, "")

        evidence_count = self.memory.count_evidence_against(pid)
        if evidence_count < self.MEETING_EVIDENCE_MIN:
            return (False, None, "")

        reason = self._build_accusation_reason(pid)
        return (True, pid, reason)

    def should_report_body(self, body_position):
        """
        Determine if the AI should report a dead body.
        AI always reports bodies (with a small reaction delay).

        Returns:
            bool: Always True when a new body is found
        """
        return True

    def decide_vote(self):
        """
        Decide who to vote for during a meeting.

        Logic:
        1. Vote for player with highest suspicion (if > 75%)
        2. If no strong suspect, skip vote
        3. Never vote for self

        Returns:
            tuple: (vote_target_id or None, reason: str)
        """
        all_scores = self.suspicion.get_all_suspicions()
        if not all_scores:
            return (None, "Not enough information to vote.")

        # Sort by suspicion, descending
        sorted_players = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)

        for pid, score in sorted_players:
            profile = self.memory.get_player_profile(pid)
            if not profile.alive or profile.is_ejected:
                continue
            if score >= self.VOTE_SUSPICION_THRESHOLD:
                if self.suspicion.should_accuse(pid):
                    reason = self._build_vote_reason(pid, score)
                    self.votes_cast.append({'target': pid, 'score': score, 'time': time.time()})
                    return (pid, reason)

        return (None, "I don't have enough evidence. Skipping this vote.")

    def _build_accusation_reason(self, player_id):
        """Build a human-readable reason for accusing a player."""
        evidence = self.memory.get_evidence_for_player(player_id)
        if not evidence:
            return "Suspicious behavior."

        reasons = []
        for e in evidence[:3]:  # Top 3 pieces of evidence
            if e.event_type == 'proximity_to_body':
                reasons.append("was seen near a dead body")
            elif e.event_type == 'alone_with_victim':
                reasons.append("was alone with the victim")
            elif e.event_type == 'near_vent':
                reasons.append("was spotted near a vent")
            elif e.event_type == 'fake_task':
                reasons.append("was faking tasks")
            elif e.event_type == 'idle_too_long':
                reasons.append("was idle for too long")
            elif e.event_type == 'running_from_body':
                reasons.append("was running from a body")
            elif e.event_type == 'direction_change':
                reasons.append("was moving erratically")
            elif e.event_type == 'contradicting_claim':
                reasons.append("contradicted their earlier statement")

        if reasons:
            return "They " + ", and ".join(reasons) + "."
        return "Suspicious behavior detected."

    def _build_vote_reason(self, player_id, score):
        """Build a reason string for voting."""
        base = self._build_accusation_reason(player_id)
        return f"Voting for this player (suspicion: {score:.0f}%). {base}"

    def on_meeting_called(self):
        """Record that a meeting was called."""
        self.last_meeting_time = time.time()
        self.meetings_called += 1

    def on_player_ejected(self, player_id):
        """Handle a player being ejected."""
        self.suspicion.reset_player(player_id)
        profile = self.memory.get_player_profile(player_id)
        profile.is_ejected = True

    def get_current_priority(self, ai_alive, tasks_remaining, highest_suspicion):
        """
        Determine current AI priority based on game state.

        Returns:
            str: Current priority action
        """
        if not ai_alive:
            return 'observe_as_ghost'

        if highest_suspicion >= 90:
            return 'call_meeting'

        if highest_suspicion >= 70:
            return 'investigate'

        if tasks_remaining > 0:
            return 'complete_tasks'

        return 'patrol'
