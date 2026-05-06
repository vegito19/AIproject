"""
AI Agent — Main Orchestrator
==============================
The central controller that coordinates all AI subsystems.
This is the brain of the AI crewmate.

Usage:
    agent = AIAgent()
    agent.initialize(game_reference)

    # Each frame:
    agent.update(dt, game_state)
    velocity = agent.get_movement()
"""

import math
import time

from ai.navigation_system import NavigationSystem
from ai.task_system import TaskSystem
from ai.observation_system import ObservationSystem
from ai.suspicion_engine import SuspicionEngine
from ai.decision_engine import DecisionEngine
from ai.memory_system import MemorySystem
from ai.chat_system import ChatSystem
from ai.rl_movement import RLMovementSystem


class AIState:
    """Enum-like class for AI behavioral states."""
    IDLE = 'idle'
    MOVING_TO_TASK = 'moving_to_task'
    DOING_TASK = 'doing_task'
    PATROLLING = 'patrolling'
    INVESTIGATING = 'investigating'
    REPORTING_BODY = 'reporting_body'
    IN_MEETING = 'in_meeting'
    FOLLOWING_GROUP = 'following_group'
    DEAD = 'dead'


class AIAgent:
    """
    Main AI Crewmate controller.

    Coordinates navigation, task completion, observation,
    suspicion analysis, decision-making, and communication.

    Design Principles:
    - Only uses observable events (no cheating)
    - Behaves like a normal crewmate
    - Decisions are explainable
    - Has realistic reaction delays
    """

    # Reaction delay range (seconds) — makes AI feel human
    REACTION_DELAY_MIN = 0.5
    REACTION_DELAY_MAX = 2.0

    # How often to re-evaluate priorities (seconds)
    PRIORITY_EVAL_INTERVAL = 3.0

    # Body report delay (seconds)
    BODY_REPORT_DELAY = 1.5

    def __init__(self):
        # Initialize all subsystems
        self.memory = MemorySystem()
        self.navigation = NavigationSystem()
        self.tasks = TaskSystem()
        self.observation = ObservationSystem(self.memory)
        self.suspicion = SuspicionEngine(self.memory)
        self.decision = DecisionEngine(self.suspicion, self.memory)
        self.chat = ChatSystem(self.memory, self.navigation)
        self.rl_movement = RLMovementSystem(self.navigation)

        # Agent state
        self.state = AIState.IDLE
        self.alive = True
        self.player_id = None
        self.position = (0, 0)
        self.velocity = (0, 0)
        self.current_room = None

        # Timing
        self.last_priority_eval = 0
        self.body_found_time = None
        self.body_found_position = None

        # Game reference
        self.game = None
        self.initialized = False

    def initialize(self, game, player_id, start_pos):
        """
        Initialize the AI agent for a new game.

        Args:
            game: Reference to the Game object
            player_id: The AI's player ID
            start_pos: (x, y) starting position
        """
        self.game = game
        self.player_id = player_id
        self.position = start_pos
        self.alive = True
        self.state = AIState.IDLE

        # Reset all subsystems
        self.memory.reset()
        self.tasks.assign_random_tasks(count=4)
        self.suspicion.reset_all()

        self.initialized = True
        print(f"[AI Agent] Initialized at position {start_pos} with ID {player_id}")

    def update(self, dt, players_data, bodies_data=None):
        """
        Main update loop — called every frame.

        Args:
            dt: Delta time since last frame
            players_data: list of player dicts
            bodies_data: list of (x, y) corpse positions
        """
        if not self.initialized or not self.alive:
            return

        now = time.time()

        # 1. Observe the world
        self.observation.observe(
            self.position[0], self.position[1],
            players_data, bodies_data
        )

        # 2. Update memory age decay
        self.memory.apply_age_decay()

        # 3. Update suspicion scores
        self.suspicion.update()

        # 4. Check for bodies
        new_body = self.observation.detect_body(
            self.position[0], self.position[1], bodies_data
        )
        if new_body and self.state != AIState.IN_MEETING:
            self.body_found_time = now
            self.body_found_position = new_body
            self.state = AIState.REPORTING_BODY

        # 5. Re-evaluate priorities periodically
        if now - self.last_priority_eval > self.PRIORITY_EVAL_INTERVAL:
            self.last_priority_eval = now
            self._evaluate_priorities()

        # 6. Execute current state behavior
        self._execute_state(dt)

        # 7. Update current room
        self.current_room = self.navigation.get_current_room(
            self.position[0], self.position[1]
        )

    def _evaluate_priorities(self):
        """Re-evaluate what the AI should be doing right now."""
        if self.state == AIState.IN_MEETING:
            return
        if self.state == AIState.REPORTING_BODY:
            return

        _, highest_score = self.suspicion.get_most_suspicious()
        tasks_left = len(self.tasks.get_pending_tasks())
        priority = self.decision.get_current_priority(
            self.alive, tasks_left, highest_score
        )

        if priority == 'call_meeting':
            should_call, target, reason = self.decision.should_call_meeting(
                is_emergency_active=getattr(self.game, 'emergency', False)
            )
            if should_call:
                self.state = AIState.INVESTIGATING
                msg = self.chat.generate_emergency_call(target, reason)
                print(f"[AI Agent] EMERGENCY: {msg}")
        elif priority == 'investigate':
            self.state = AIState.INVESTIGATING
        elif priority == 'complete_tasks':
            if self.state != AIState.DOING_TASK:
                self.state = AIState.MOVING_TO_TASK
        elif priority == 'patrol':
            self.state = AIState.PATROLLING

    def _execute_state(self, dt):
        """Execute behavior based on current state."""
        if self.state == AIState.IDLE:
            self._do_idle(dt)
        elif self.state == AIState.MOVING_TO_TASK:
            self._do_move_to_task(dt)
        elif self.state == AIState.DOING_TASK:
            self._do_task(dt)
        elif self.state == AIState.PATROLLING:
            self._do_patrol(dt)
        elif self.state == AIState.INVESTIGATING:
            self._do_investigate(dt)
        elif self.state == AIState.REPORTING_BODY:
            self._do_report_body(dt)
        elif self.state == AIState.IN_MEETING:
            self._do_meeting(dt)

    def _do_idle(self, dt):
        """Idle state — find something to do."""
        self.velocity = (0, 0)
        if not self.tasks.all_tasks_done():
            self.state = AIState.MOVING_TO_TASK
        else:
            self.state = AIState.PATROLLING

    def _do_move_to_task(self, dt):
        """Navigate toward the next task location."""
        task_loc = self.tasks.get_current_task_location()
        if not task_loc:
            self.state = AIState.PATROLLING
            return

        if not self.navigation.is_moving:
            next_task = self.tasks.get_next_task()
            if next_task:
                self.navigation.set_destination(
                    self.position[0], self.position[1],
                    next_task.location[0], next_task.location[1]
                )

        self.velocity = self.rl_movement.get_movement(
            self.position, (next_task.location[0], next_task.location[1])
        )

        if self.navigation.has_reached_destination():
            next_task = self.tasks.get_next_task()
            if next_task:
                self.tasks.start_task(next_task)
                self.state = AIState.DOING_TASK
                self.velocity = (0, 0)

    def _do_task(self, dt):
        """Simulate performing a task (standing still for the duration)."""
        self.velocity = (0, 0)
        completed = self.tasks.update()
        if completed:
            self.memory.log_event(
                'task_completed', self.player_id, self.position,
                details={'task': completed.name},
                confidence=1.0
            )
            print(f"[AI Agent] Completed task: {completed.name}")
            if self.tasks.all_tasks_done():
                self.state = AIState.PATROLLING
            else:
                self.state = AIState.MOVING_TO_TASK

    def _do_patrol(self, dt):
        """Roam the map, observing players."""
        if self.navigation.has_reached_destination():
            self.navigation.set_random_destination(
                self.position[0], self.position[1]
            )
        
        if self.navigation.current_destination:
            self.velocity = self.rl_movement.get_movement(
                self.position, self.navigation.current_destination
            )
        else:
            self.velocity = (0, 0)

    def _do_investigate(self, dt):
        """Move toward the most suspicious player's last known location."""
        pid, score = self.suspicion.get_most_suspicious()
        if pid:
            profile = self.memory.get_player_profile(pid)
            if profile.last_known_position:
                if not self.navigation.is_moving:
                    self.navigation.set_destination(
                        self.position[0], self.position[1],
                        profile.last_known_position[0],
                        profile.last_known_position[1]
                    )
        else:
            self.state = AIState.PATROLLING
            return

        self.velocity = self.rl_movement.get_movement(
            self.position, profile.last_known_position
        )

    def _do_report_body(self, dt):
        """Handle body discovery with realistic reaction delay."""
        self.velocity = (0, 0)
        now = time.time()

        if self.body_found_time and now - self.body_found_time >= self.BODY_REPORT_DELAY:
            room = self.navigation.get_current_room(
                self.body_found_position[0], self.body_found_position[1]
            ) if self.body_found_position else None
            msg = self.chat.generate_body_report(room)
            print(f"[AI Agent] REPORT: {msg}")

            # Check who was near the body
            self._analyze_body_scene()

            self.body_found_time = None
            self.body_found_position = None
            self.state = AIState.IN_MEETING

    def _analyze_body_scene(self):
        """Analyze who was near the body when it was discovered."""
        if not self.body_found_position:
            return
        for pid, profile in self.memory.player_profiles.items():
            if pid == self.player_id:
                continue
            if profile.last_known_position:

                dist = math.sqrt(
                    (profile.last_known_position[0] - self.body_found_position[0])**2 +
                    (profile.last_known_position[1] - self.body_found_position[1])**2
                )
                if dist < 200:
                    self.memory.log_event(
                        'proximity_to_body', pid, profile.last_known_position,
                        details={'body_pos': self.body_found_position, 'distance': dist},
                        confidence=0.85
                    )

    def _do_meeting(self, dt):
        """Handle behavior during a meeting."""
        self.velocity = (0, 0)

    def on_meeting_start(self):
        """Called when a meeting begins."""
        self.state = AIState.IN_MEETING
        self.decision.on_meeting_called()

    def on_meeting_vote(self):
        """Called when it's time to vote."""
        target, reason = self.decision.decide_vote()
        if target:
            msg = self.chat.generate_vote_message(target, reason)
        else:
            msg = self.chat.generate_skip_message()
        print(f"[AI Agent] VOTE: {msg}")
        return target

    def on_meeting_end(self, ejected_player=None):
        """Called when a meeting ends."""
        if ejected_player:
            self.decision.on_player_ejected(ejected_player)
        if not self.tasks.all_tasks_done():
            self.state = AIState.MOVING_TO_TASK
        else:
            self.state = AIState.PATROLLING

    def on_death(self):
        """Called when the AI is killed."""
        self.alive = False
        self.state = AIState.DEAD
        self.velocity = (0, 0)
        print("[AI Agent] I have been killed!")

    def get_movement(self):
        """
        Get the current velocity vector for movement.

        Returns:
            tuple: (vel_x, vel_y)
        """
        return self.velocity

    def get_state_info(self):
        """Get a debug summary of the AI's current state."""
        pid, score = self.suspicion.get_most_suspicious()
        return {
            'state': self.state,
            'position': self.position,
            'room': self.current_room,
            'alive': self.alive,
            'tasks_done': self.tasks.tasks_completed,
            'tasks_pending': len(self.tasks.get_pending_tasks()),
            'most_suspicious': (pid, f"{score:.1f}%") if pid else None,
            'total_events': len(self.memory.events),
        }
