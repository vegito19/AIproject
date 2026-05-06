"""
Chat System — LLM-Powered Dialogue Generation
===============================================
Generates conversational responses during meetings.
Attempts to use a local Phi-3 LLM (via Ollama) to generate natural,
human-like chat messages.
Automatically falls back to rule-based template strings if the LLM
is unavailable or times out.
"""

import random
import requests

class ChatSystem:
    """Dialogue generation for meeting communication with LLM integration."""

    # --- Dialogue Templates (Fallbacks) ---
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
        
        # LLM Settings for Qwen via Ollama
        self.use_llm = True
        self.llm_url = "http://localhost:11434/api/generate"
        self.llm_model = "qwen3.5:4b"
        self.llm_timeout = 2.0  # Fast timeout so the game doesn't freeze

    def _query_llm(self, context_prompt, fallback_text):
        """Query the local Phi-3 model, fallback to template if it fails."""
        if not self.use_llm:
            return fallback_text
            
        system_prompt = (
            "You are a crewmate in a game of Among Us. "
            "Write a very short, casual chat message in response to the prompt. "
            "Keep it under 15 words. Sound like a real human gamer. "
            "Do not use quotes or roleplay actions."
        )
        
        try:
            response = requests.post(
                self.llm_url,
                json={
                    "model": self.llm_model,
                    "system": system_prompt,
                    "prompt": context_prompt,
                    "stream": False
                },
                timeout=self.llm_timeout
            )
            if response.status_code == 200:
                result = response.json().get("response", fallback_text).strip()
                # Clean up quotes if the model added them anyway
                return result.strip('"').strip("'")
        except requests.exceptions.RequestException:
            # If Ollama isn't running or times out, fallback to rules and disable LLM to prevent future freezes
            print("[ChatSystem] Local Phi-3 LLM not reachable or too slow. Falling back to rule-based templates.")
            self.use_llm = False
            
        return fallback_text

    def generate_accusation(self, target_id, reason="", room=None):
        """Generate an accusation message."""
        player_name = self._get_player_name(target_id)
        if not room:
            profile = self.memory.get_player_profile(target_id)
            if profile.last_known_position:
                room = self.navigation.get_current_room(*profile.last_known_position)
        room = room or "an unknown area"
        
        fallback = random.choice(self.ACCUSE_TEMPLATES).format(
            player=player_name, room=room, reason=reason
        )
        prompt = f"Accuse {player_name} of being the impostor. You saw them in {room}. Context: {reason}"
        
        msg = self._query_llm(prompt, fallback)
        self.chat_history.append(msg)
        return msg

    def generate_defense(self, ai_room=None, ai_task=None, tasks_done=0):
        """Generate a self-defense message."""
        room = ai_room or "my task area"
        task = ai_task or "my assigned tasks"
        
        fallback = random.choice(self.DEFEND_TEMPLATES).format(
            room=room, task=task, count=tasks_done
        )
        prompt = f"Defend yourself. Tell everyone you were in {room} doing {task}. You have finished {tasks_done} tasks."
        
        msg = self._query_llm(prompt, fallback)
        self.chat_history.append(msg)
        return msg

    def generate_support(self, player_id, room=None):
        """Generate a message supporting another player."""
        player_name = self._get_player_name(player_id)
        room = room or "the area"
        
        fallback = random.choice(self.SUPPORT_TEMPLATES).format(
            player=player_name, room=room
        )
        prompt = f"Vouch for {player_name}. You saw them in {room} doing tasks so they are safe."
        
        msg = self._query_llm(prompt, fallback)
        self.chat_history.append(msg)
        return msg

    def generate_uncertain(self, player_id=None, room=None):
        """Generate an uncertain/hesitant message."""
        player_name = self._get_player_name(player_id) if player_id else "someone"
        room = room or "some area"
        
        fallback = random.choice(self.UNCERTAIN_TEMPLATES).format(
            player=player_name, room=room
        )
        prompt = f"Say you are not sure who the impostor is, but {player_name} was acting weird near {room}."
        
        msg = self._query_llm(prompt, fallback)
        self.chat_history.append(msg)
        return msg

    def generate_vote_message(self, target_id, reason=""):
        """Generate a voting announcement."""
        player_name = self._get_player_name(target_id)
        
        fallback = random.choice(self.VOTE_TEMPLATES).format(
            player=player_name, reason=reason
        )
        prompt = f"Announce that you are voting for {player_name} because {reason}."
        
        msg = self._query_llm(prompt, fallback)
        self.chat_history.append(msg)
        return msg

    def generate_skip_message(self):
        """Generate a skip vote message."""
        fallback = random.choice(self.SKIP_TEMPLATES)
        prompt = "Announce that you are skipping the vote because you don't have enough evidence."
        
        msg = self._query_llm(prompt, fallback)
        self.chat_history.append(msg)
        return msg

    def generate_body_report(self, room=None):
        """Generate a body report message."""
        room = room or "an unknown location"
        
        fallback = random.choice(self.REPORT_TEMPLATES).format(room=room)
        prompt = f"You just found a dead body in {room}. Call it out in panic."
        
        msg = self._query_llm(prompt, fallback)
        self.chat_history.append(msg)
        return msg

    def generate_emergency_call(self, target_id, reason=""):
        """Generate an emergency meeting call message."""
        player_name = self._get_player_name(target_id)
        
        fallback = random.choice(self.EMERGENCY_TEMPLATES).format(
            player=player_name, reason=reason
        )
        prompt = f"You just called an emergency meeting to accuse {player_name}. Reason: {reason}."
        
        msg = self._query_llm(prompt, fallback)
        self.chat_history.append(msg)
        return msg

    def generate_meeting_summary(self):
        """Generate a comprehensive statement for a meeting."""
        statements = []
        all_scores = {}
        for pid in self.memory.player_profiles:
            evidence = self.memory.get_evidence_for_player(pid)
            if evidence:
                all_scores[pid] = len(evidence)

        if not all_scores:
            return self.generate_uncertain()

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
                statements.append(f"Player {self._get_player_name(pid)}: {reason}{room_text}")

        fallback = " | ".join(statements) if statements else self.generate_uncertain()
        
        prompt = f"Summarize your suspicions in one short sentence based on this evidence: {fallback}"
        msg = self._query_llm(prompt, fallback)
        return msg

    def _event_to_text(self, event):
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
        return f"Player {player_id}"

    def get_recent_messages(self, count=5):
        return self.chat_history[-count:]
