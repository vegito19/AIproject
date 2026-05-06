"""
Task System — Task Completion Logic
=====================================
Manages the AI's task queue and simulates task completion.
The AI navigates to task locations and "performs" them with
realistic timing (not instant).
"""

import random
import time


class TaskInfo:
    """Information about a single task."""
    def __init__(self, name, location, duration, room):
        self.name = name
        self.location = location   # (x, y) coordinates
        self.duration = duration   # seconds to complete
        self.room = room
        self.completed = False
        self.in_progress = False
        self.start_time = None


class TaskSystem:
    """Manages AI task assignment and completion."""

    def __init__(self):
        self.all_tasks = []
        self.assigned_tasks = []
        self.current_task = None
        self.tasks_completed = 0
        self._define_tasks()

    def _define_tasks(self):
        """Define all available tasks from the game map."""
        self.all_tasks = [
            TaskInfo("Stabilize Navigation", (5610, 1290), 4.0, "cockpit"),
            TaskInfo("Empty Garbage", (3940, 321), 3.0, "cafeteria"),
            TaskInfo("Reboot Wifi", (3700, 1554), 5.0, "admin"),
            TaskInfo("Fix Electricity Wires", (3166, 1846), 4.5, "electrical"),
            TaskInfo("Divert Power to Reactor", (1031, 1216), 3.5, "reactor"),
            TaskInfo("Align Engine Output", (1117, 837), 3.0, "upper_engine"),
            TaskInfo("Fuel Lower Engine", (1226, 2300), 4.0, "lower_engine"),
            TaskInfo("Clear Asteroids", (4513, 450), 6.0, "weapons"),
        ]

    def assign_random_tasks(self, count=4):
        """Assign a random subset of tasks to the AI."""
        available = [t for t in self.all_tasks if not t.completed]
        self.assigned_tasks = random.sample(available, min(count, len(available)))
        self.current_task = None

    def get_next_task(self):
        """Get the next uncompleted task."""
        for task in self.assigned_tasks:
            if not task.completed and not task.in_progress:
                return task
        return None

    def start_task(self, task):
        """Begin working on a task."""
        task.in_progress = True
        task.start_time = time.time()
        self.current_task = task

    def update(self):
        """Check if current task is done (enough time has passed)."""
        if self.current_task and self.current_task.in_progress:
            elapsed = time.time() - self.current_task.start_time
            if elapsed >= self.current_task.duration:
                self.current_task.completed = True
                self.current_task.in_progress = False
                self.tasks_completed += 1
                completed = self.current_task
                self.current_task = None
                return completed
        return None

    def is_doing_task(self):
        return self.current_task is not None and self.current_task.in_progress

    def get_current_task_location(self):
        if self.current_task:
            return self.current_task.location
        next_t = self.get_next_task()
        if next_t:
            return next_t.location
        return None

    def all_tasks_done(self):
        return all(t.completed for t in self.assigned_tasks)

    def get_completed_tasks(self):
        return [t for t in self.assigned_tasks if t.completed]

    def get_pending_tasks(self):
        return [t for t in self.assigned_tasks if not t.completed]
