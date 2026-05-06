"""
AI Bot — Integration Bridge
==============================
Extends the existing Bot sprite class to be controlled by the AIAgent.
This is the bridge between the AI system and the game engine.

The AIBot replaces the dumb Bot in freeplay mode, making it
move intelligently, observe players, track suspicion, and
participate in meetings/voting.
"""

import pygame as pg
from sprites import Bot
from ai.ai_agent import AIAgent

vec = pg.math.Vector2


class AIBot(Bot):
    """
    An AI-controlled Bot that uses the full AI agent system
    for navigation, observation, suspicion, and decision-making.

    Inherits from Bot to maintain full compatibility with the
    existing sprite/collision/rendering system.
    """

    def __init__(self, game, x, y, bot_direction, bot_type, bot_colour, bot_id=None):
        """
        Initialize the AIBot with all standard Bot parameters
        plus the AI agent.

        Args:
            game: Game reference
            x, y: Spawn position
            bot_direction: Initial facing direction
            bot_type: Bot type identifier (e.g. 'bot1')
            bot_colour: Colour string (e.g. 'Red')
            bot_id: Unique ID for the AI agent (defaults to hash of bot_type)
        """
        super().__init__(game, x, y, bot_direction, bot_type, bot_colour)

        # Create the AI agent
        self.ai_agent = AIAgent()
        self.bot_id = bot_id or hash(bot_type) % 1000
        self.ai_initialized = False

        # Store colour-specific sprite lists for animation
        self._setup_animation_sprites(bot_colour)

        # Track frame count for periodic AI debug output
        self._frame_count = 0
        self._debug_interval = 300  # Print debug every 300 frames (~5 sec at 60fps)

    def _setup_animation_sprites(self, colour):
        """Store references to directional sprite lists for walk animation."""
        from settings import (
            red_player_imgs_left, red_player_imgs_right,
            red_player_imgs_up, red_player_imgs_down,
            blue_player_imgs_left, blue_player_imgs_right,
            blue_player_imgs_up, blue_player_imgs_down,
            orange_player_imgs_left, orange_player_imgs_right,
            orange_player_imgs_up, orange_player_imgs_down,
            yellow_player_imgs_left, yellow_player_imgs_right,
            yellow_player_imgs_up, yellow_player_imgs_down,
            green_player_imgs_left, green_player_imgs_right,
            green_player_imgs_up, green_player_imgs_down,
            black_player_imgs_left, black_player_imgs_right,
            black_player_imgs_up, black_player_imgs_down,
            brown_player_imgs_left, brown_player_imgs_right,
            brown_player_imgs_up, brown_player_imgs_down,
            pink_player_imgs_left, pink_player_imgs_right,
            pink_player_imgs_up, pink_player_imgs_down,
            purple_player_imgs_left, purple_player_imgs_right,
            purple_player_imgs_up, purple_player_imgs_down,
            white_player_imgs_left, white_player_imgs_right,
            white_player_imgs_up, white_player_imgs_down,
        )

        colour_map = {
            "Red":    (red_player_imgs_left, red_player_imgs_right, red_player_imgs_up, red_player_imgs_down),
            "Blue":   (blue_player_imgs_left, blue_player_imgs_right, blue_player_imgs_up, blue_player_imgs_down),
            "Orange": (orange_player_imgs_left, orange_player_imgs_right, orange_player_imgs_up, orange_player_imgs_down),
            "Yellow": (yellow_player_imgs_left, yellow_player_imgs_right, yellow_player_imgs_up, yellow_player_imgs_down),
            "Green":  (green_player_imgs_left, green_player_imgs_right, green_player_imgs_up, green_player_imgs_down),
            "Black":  (black_player_imgs_left, black_player_imgs_right, black_player_imgs_up, black_player_imgs_down),
            "Brown":  (brown_player_imgs_left, brown_player_imgs_right, brown_player_imgs_up, brown_player_imgs_down),
            "Pink":   (pink_player_imgs_left, pink_player_imgs_right, pink_player_imgs_up, pink_player_imgs_down),
            "Purple": (purple_player_imgs_left, purple_player_imgs_right, purple_player_imgs_up, purple_player_imgs_down),
            "White":  (white_player_imgs_left, white_player_imgs_right, white_player_imgs_up, white_player_imgs_down),
        }

        if colour in colour_map:
            self.anim_left, self.anim_right, self.anim_up, self.anim_down = colour_map[colour]
        else:
            # Fallback to red
            self.anim_left, self.anim_right, self.anim_up, self.anim_down = colour_map["Red"]

        self.anim_index = 0
        self.anim_timer = 0
        self.anim_speed = 150  # ms between animation frames

    def initialize_ai(self):
        """
        Initialize the AI agent. Call this after the game
        has fully set up (after new() and player creation).
        """
        self.ai_agent.initialize(
            game=self.game,
            player_id=self.bot_id,
            start_pos=(self.pos.x, self.pos.y)
        )
        self.ai_initialized = True

    def update(self):
        """
        Override Bot.update() to inject AI-driven velocity
        before the physics/collision step.
        """
        if not self.alive_status:
            # Dead bots don't move
            self.vel = vec(0, 0)
            super().update()
            return

        if not self.ai_initialized:
            self.initialize_ai()

        # --- Gather observable game state for the AI ---
        players_data = self._gather_player_data()
        bodies_data = self._gather_body_data()

        # --- Update the AI agent ---
        dt = self.game.dt
        self.ai_agent.update(dt, players_data, bodies_data)

        # --- Apply AI movement to sprite velocity ---
        ai_vel = self.ai_agent.get_movement()
        self.vel = vec(ai_vel[0], ai_vel[1])

        # --- Update AI's internal position (for pathfinding) ---
        self.ai_agent.position = (self.pos.x, self.pos.y)

        # --- Animate the sprite based on movement direction ---
        self._animate_walk()

        # --- Run the standard Bot physics (pos += vel * dt, collisions) ---
        super().update()

        # --- Periodic debug output ---
        self._frame_count += 1
        if self._frame_count % self._debug_interval == 0:
            self._print_debug()

    def _gather_player_data(self):
        """
        Collect observable player data from the game.
        Only includes what the AI could actually see.
        """
        players = []

        # The human player
        if hasattr(self.game, 'player') and self.game.player:
            p = self.game.player
            players.append({
                'id': getattr(p, 'player_id', 0),
                'x': p.pos.x,
                'y': p.pos.y,
                'alive': p.alive_status,
                'colour': p.player_colour,
                'tasks_completed': getattr(p, 'tasks_completed', 0),
            })

        # Other bots (observed as "other players")
        for bot in self.game.bots:
            if bot is self:
                continue  # Don't observe yourself
            players.append({
                'id': hash(bot.type) % 1000,
                'x': bot.pos.x,
                'y': bot.pos.y,
                'alive': bot.alive_status,
                'colour': bot.bot_colour,
                'tasks_completed': 0,
            })

        # Multiplayer players (if any)
        if hasattr(self.game, 'Players'):
            for pid, p in self.game.Players.items():
                if pid == self.bot_id:
                    continue
                players.append({
                    'id': pid,
                    'x': p.pos.x,
                    'y': p.pos.y,
                    'alive': p.alive_status,
                    'colour': getattr(p, 'player_colour', 'Unknown'),
                    'tasks_completed': getattr(p, 'tasks_completed', 0),
                })

        return players

    def _gather_body_data(self):
        """
        Collect dead body positions from the game.
        Only includes visible corpses.
        """
        bodies = []

        # Dead bots = bodies on the ground
        for bot in self.game.bots:
            if bot is self:
                continue
            if not bot.alive_status:
                bodies.append((bot.pos.x, bot.pos.y))

        # Dead human player
        if hasattr(self.game, 'player') and self.game.player:
            if not self.game.player.alive_status:
                corpse_pos = getattr(self.game.player, 'pos_corpse', self.game.player.pos)
                bodies.append((corpse_pos.x, corpse_pos.y))

        return bodies

    def _animate_walk(self):
        """Animate the bot sprite based on current velocity direction."""
        if self.vel.x == 0 and self.vel.y == 0:
            # Standing still — use first frame of current direction
            return

        now = pg.time.get_ticks()
        if now - self.anim_timer < self.anim_speed:
            return
        self.anim_timer = now

        # Determine primary direction
        if abs(self.vel.x) >= abs(self.vel.y):
            if self.vel.x > 0:
                sprites = self.anim_right
            else:
                sprites = self.anim_left
        else:
            if self.vel.y > 0:
                sprites = self.anim_down
            else:
                sprites = self.anim_up

        self.anim_index = (self.anim_index + 1) % len(sprites)
        self.image = sprites[self.anim_index]

    def on_killed(self):
        """Called when this bot is killed by the impostor."""
        self.alive_status = False
        self.image = self.dead_player_img
        self.ai_agent.on_death()

    def on_meeting_start(self):
        """Called when a meeting starts."""
        if self.ai_initialized:
            self.ai_agent.on_meeting_start()

    def on_meeting_end(self, ejected_id=None):
        """Called when a meeting ends."""
        if self.ai_initialized:
            self.ai_agent.on_meeting_end(ejected_id)

    def _print_debug(self):
        """Print AI debug info periodically."""
        if not self.ai_initialized:
            return
        info = self.ai_agent.get_state_info()
        print(f"[AIBot {self.bot_colour}] "
              f"State: {info['state']} | "
              f"Room: {info['room']} | "
              f"Tasks: {info['tasks_done']}/{info['tasks_done'] + info['tasks_pending']} | "
              f"Pos: ({self.pos.x:.0f}, {self.pos.y:.0f})")
