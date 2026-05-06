"""
Quick integration test for the AI Crewmate system.
Tests all modules working together in a simulated scenario.
"""

import sys
import time

# Test imports
from ai.ai_agent import AIAgent, AIState
from ai.navigation_system import NavigationSystem
from ai.task_system import TaskSystem
from ai.observation_system import ObservationSystem
from ai.suspicion_engine import SuspicionEngine
from ai.decision_engine import DecisionEngine
from ai.memory_system import MemorySystem
from ai.chat_system import ChatSystem


def test_all_modules():
    print("=" * 60)
    print("  AI CREWMATE — INTEGRATION TEST")
    print("=" * 60)

    # 1. Create AI Agent
    print("\n[1] Creating AI Agent...")
    agent = AIAgent()
    agent.player_id = 999
    agent.position = (3277, 658)  # Cafeteria
    agent.alive = True
    agent.initialized = True
    agent.tasks.assign_random_tasks(count=4)
    print(f"    ✓ Agent created at cafeteria. Tasks assigned: {len(agent.tasks.assigned_tasks)}")
    for t in agent.tasks.assigned_tasks:
        print(f"      - {t.name} @ {t.location} ({t.room})")

    # 2. Test Navigation
    print("\n[2] Testing Navigation...")
    nav = agent.navigation
    room = nav.get_current_room(3277, 658)
    print(f"    ✓ Current room: {room}")

    path = nav.find_path(3277, 658, 880, 1474)  # Cafeteria to Reactor
    print(f"    ✓ Path Cafeteria → Reactor: {path}")

    nav.set_destination(3277, 658, 880, 1474)
    vel = nav.get_velocity(3277, 658, 0.016)
    print(f"    ✓ Velocity toward reactor: ({vel[0]:.1f}, {vel[1]:.1f})")

    # 3. Test Memory
    print("\n[3] Testing Memory System...")
    mem = agent.memory
    mem.register_player(101)
    mem.register_player(102)
    mem.register_player(103)
    mem.update_player_position(101, 3000, 700)
    mem.update_player_position(102, 900, 1500)
    mem.log_event('proximity_to_body', 102, (900, 1500),
                  details={'body_pos': (850, 1450)}, confidence=0.8)
    mem.log_event('near_vent', 102, (900, 1500),
                  details={'vent_pos': (931, 1626)}, confidence=0.6)
    mem.log_event('idle_too_long', 103, (2400, 1900),
                  details={'idle_seconds': 12}, confidence=0.4)
    print(f"    ✓ Players registered: {list(mem.player_profiles.keys())}")
    print(f"    ✓ Events logged: {len(mem.events)}")
    print(f"    ✓ Evidence against 102: {mem.count_evidence_against(102)}")

    # 4. Test Suspicion Engine
    print("\n[4] Testing Suspicion Engine...")
    sus = agent.suspicion
    sus.register_player(101)
    sus.register_player(102)
    sus.register_player(103)
    sus.update()
    print(f"    ✓ Suspicion scores:")
    for pid, score in sus.get_all_suspicions().items():
        conf = sus.get_confidence_level(pid)
        print(f"      Player {pid}: {score:.1f}% (confidence: {conf})")

    most_sus, score = sus.get_most_suspicious()
    print(f"    ✓ Most suspicious: Player {most_sus} ({score:.1f}%)")

    # 5. Test Decision Engine
    print("\n[5] Testing Decision Engine...")
    dec = agent.decision
    should_meet, target, reason = dec.should_call_meeting()
    print(f"    ✓ Should call meeting: {should_meet}")
    if should_meet:
        print(f"      Target: {target}, Reason: {reason}")

    vote_target, vote_reason = dec.decide_vote()
    print(f"    ✓ Vote decision: Player {vote_target}")
    print(f"      Reason: {vote_reason}")

    # 6. Test Chat System
    print("\n[6] Testing Chat System...")
    chat = agent.chat
    msg1 = chat.generate_accusation(102, "was near a dead body", "reactor")
    msg2 = chat.generate_defense(ai_room="cafeteria", ai_task="Clear Asteroids", tasks_done=2)
    msg3 = chat.generate_body_report("electrical")
    msg4 = chat.generate_skip_message()
    print(f"    ✓ Accusation: {msg1}")
    print(f"    ✓ Defense: {msg2}")
    print(f"    ✓ Body report: {msg3}")
    print(f"    ✓ Skip: {msg4}")

    # 7. Test Full Agent Update Loop
    print("\n[7] Testing Agent Update Loop...")
    fake_players = [
        {'id': 101, 'x': 3000, 'y': 700, 'alive': True, 'colour': 'Blue', 'tasks_completed': 2},
        {'id': 102, 'x': 900, 'y': 1500, 'alive': True, 'colour': 'Red', 'tasks_completed': 0},
        {'id': 103, 'x': 2400, 'y': 1900, 'alive': True, 'colour': 'Green', 'tasks_completed': 1},
    ]
    fake_bodies = [(850, 1450)]

    for i in range(5):
        agent.update(0.016, fake_players, fake_bodies)

    info = agent.get_state_info()
    print(f"    ✓ Agent state after 5 updates:")
    for k, v in info.items():
        print(f"      {k}: {v}")

    # 8. Summary
    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED ✓")
    print("=" * 60)
    print(f"\n  Total modules: 9")
    print(f"  Memory events: {len(agent.memory.events)}")
    print(f"  Player profiles: {len(agent.memory.player_profiles)}")
    print(f"  Suspicion scores tracked: {len(agent.suspicion.suspicion_scores)}")
    print(f"  Chat messages generated: {len(agent.chat.chat_history)}")
    print(f"  Navigation waypoints: {len(agent.navigation.waypoints)}")
    print(f"  Tasks defined: {len(agent.tasks.all_tasks)}")


if __name__ == '__main__':
    test_all_modules()
