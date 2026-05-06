"""
Chat System — Rule-Based Dialogue Generation
===============================================
Generates conversational responses during meetings.
Uses template strings with evidence-driven variable substitution.

Phase 1: Rule-based (no LLM needed).
"""

import random


class ChatSystem:
    """Rule-based dialogue generation for meeting communication."""

    # --- Dialogue Templates ---
    ACCUSE_TEMPLATES = [
        "I saw {player} near the body in {room}.",
        "I think {player} is suspicious. {reason}",
        "{player} was acting weird near {room}. We should vote them.",
        "I'm pretty sure it's {player}. {reason}",
        "Has anyone else noticed {player}? {reason}",
    ]

    DEFEND_TEMPLATES = [
        "I was in {room} doing {task}. I'm not the impostor.",
        "I was with the group. You can check my tasks.",
        "I've completed {count} tasks already. Why would I be sus?",
        "I was nowhere near the body. I was in {room}.",
        "That's not true. I can prove I was in {room}.",
    ]

    SUPPORT_TEMPLATES = [
        "I can confirm {player} was with me in {room}.",
        "I saw {player} doing tasks. They seem safe.",
        "{player} reported the body quickly. That's a good sign.",
        "I don't think it's {player}. They've been doing tasks.",
    ]

    UNCERTAIN_TEMPLATES = [
        "I'm not sure, but {player} was acting odd near {room}.",
        "I don't have enough evidence to accuse anyone.",
        "Something felt off about {player}, but I can't be certain.",
        "I need more time to observe before making a call.",
    ]

    VOTE_TEMPLATES = [
        "I'm voting for {player}. {reason}",
        "My vote goes to {player}. {reason}",
        "Based on what I've seen, I'm voting {player}.",
    ]

    SKIP_TEMPLATES = [
        "I don't have enough evidence. Skipping this vote.",
        "I'm not confident enough to vote anyone. Skipping.",
        "Let's wait and gather more evidence. I'm skipping.",
        "No strong suspects. I'll skip this round.",
    ]

    REPORT_TEMPLATES = [
        "I found a body in {room}! Everyone come quick!",
        "Dead body in {room}! Who was near there?",
        "Body found near {room}! I just walked in and saw it.",
    ]

    EMERGENCY_TEMPLATES = [
        "I'm calling an emergency meeting. I have evidence against {player}.",
        "Emergency! I think {player} is the impostor. {reason}",
        "We need to discuss {player}. I've seen suspicious behavior.",
    ]

    def __init__(self, memory_system, navigation_system):
        self.memory = memory_system
        self.navigation = navigation_system
        self.chat_history = []

    def generate_accusation(self, target_id, reason="", room=None):
        """Generate an accusation message."""
        player_name = self._get_player_name(target_id)
        if not room:
            profile = self.memory.get_player_profile(target_id)
            if profile.last_known_position:
                room = self.navigation.get_current_room(*profile.last_known_position)
        room = room or "an unknown area"
        msg = random.choice(self.ACCUSE_TEMPLATES).format(
            player=player_name, room=room, reason=reason
        )
        self.chat_history.append(msg)
        return msg

    def generate_defense(self, ai_room=None, ai_task=None, tasks_done=0):
        """Generate a self-defense message."""
        room = ai_room or "my task area"
        task = ai_task or "my assigned tasks"
        msg = random.choice(self.DEFEND_TEMPLATES).format(
            room=room, task=task, count=tasks_done
        )
        self.chat_history.append(msg)
        return msg

    def generate_support(self, player_id, room=None):
        """Generate a message supporting another player."""
        player_name = self._get_player_name(player_id)
        room = room or "the area"
        msg = random.choice(self.SUPPORT_TEMPLATES).format(
            player=player_name, room=room
        )
        self.chat_history.append(msg)
        return msg

    def generate_uncertain(self, player_id=None, room=None):
        """Generate an uncertain/hesitant message."""
        player_name = self._get_player_name(player_id) if player_id else "someone"
        room = room or "some area"
        msg = random.choice(self.UNCERTAIN_TEMPLATES).format(
            player=player_name, room=room
        )
        self.chat_history.append(msg)
        return msg

    def generate_vote_message(self, target_id, reason=""):
        """Generate a voting announcement."""
        player_name = self._get_player_name(target_id)
        msg = random.choice(self.VOTE_TEMPLATES).format(
            player=player_name, reason=reason
        )
        self.chat_history.append(msg)
        return msg

    def generate_skip_message(self):
        """Generate a skip vote message."""
        msg = random.choice(self.SKIP_TEMPLATES)
        self.chat_history.append(msg)
        return msg

    def generate_body_report(self, room=None):
        """Generate a body report message."""
        room = room or "an unknown location"
        msg = random.choice(self.REPORT_TEMPLATES).format(room=room)
        self.chat_history.append(msg)
        return msg

    def generate_emergency_call(self, target_id, reason=""):
        """Generate an emergency meeting call message."""
        player_name = self._get_player_name(target_id)
        msg = random.choice(self.EMERGENCY_TEMPLATES).format(
            player=player_name, reason=reason
        )
        self.chat_history.append(msg)
        return msg

    def generate_meeting_summary(self):
        """
        Generate a comprehensive statement for a meeting,
        combining observations and suspicions.
        """
        statements = []
        all_scores = {}
        for pid in self.memory.player_profiles:
            evidence = self.memory.get_evidence_for_player(pid)
            if evidence:
                all_scores[pid] = len(evidence)

        if not all_scores:
            return self.generate_uncertain()

        # Report on top 2 most suspicious
        sorted_players = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        for pid, count in sorted_players[:2]:
            evidence = self.memory.get_evidence_for_player(pid)
            if evidence:
                top_event = evidence[0]
                room = self.navigation.get_current_room(
                    *top_event.location
                ) if top_event.location else None
                reason = self._event_to_text(top_event)
                room_text = f" in {room}" if room else ""
                statements.append(
                    f"Player {self._get_player_name(pid)}: {reason}{room_text}"
                )

        return " | ".join(statements) if statements else self.generate_uncertain()

    def _event_to_text(self, event):
        """Convert a memory event to human-readable text."""
        texts = {
            'proximity_to_body': "was seen near a dead body",
            'running_from_body': "was running away from a body",
            'fake_task': "appeared to be faking tasks",
            'idle_too_long': "was standing idle suspiciously",
            'near_vent': "was spotted near a vent",
            'direction_change': "was moving erratically",
            'alone_with_victim': "was alone with the victim before the kill",
            'contradicting_claim': "contradicted their earlier statement",
        }
        return texts.get(event.event_type, "was acting suspiciously")

    def _get_player_name(self, player_id):
        """Get a display name for a player."""
        profile = self.memory.get_player_profile(player_id)
        if profile:
            return f"Player {player_id}"
        return f"Player {player_id}"

    def get_recent_messages(self, count=5):
        """Get the last N messages generated by the AI."""
        return self.chat_history[-count:]
