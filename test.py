"""
tests/test_alerts.py

Manual test for AlertManager.

Run:

python tests/test_alerts.py
"""

import time

from ai.rules import RuleEngine
from ai.alerts import AlertManager


# -------------------------------------------------------------
# Desk Map
# -------------------------------------------------------------

desk_map = {
    1: [
        (100, 100),
        (300, 100),
        (300, 250),
        (100, 250),
    ]
}


# -------------------------------------------------------------
# Components
# -------------------------------------------------------------

engine = RuleEngine(desk_map)

alerts = AlertManager()


# -------------------------------------------------------------
# Student
# -------------------------------------------------------------

student = {
    "student_id": 1,
    "bbox": (100, 100, 200, 300),
    "left_wrist": (150, 180),
    "right_wrist": (200, 180),
    "left_shoulder": (130, 150),
    "right_shoulder": (170, 150),
    "nose": (150, 120),
    "left_eye": (142, 116),
    "right_eye": (158, 116),
    "left_ear": (134, 122),
    "right_ear": (166, 122),
    "yaw": 0.0,
    "pitch": 0.0,
    "roll": 0.0,
    "desk_id": 1,
}

students = [student]


# -------------------------------------------------------------
# Normal
# -------------------------------------------------------------

print("\nNORMAL")

results = engine.update(student, students)

print(alerts.update(student["student_id"], results))


# -------------------------------------------------------------
# Hands Off
# -------------------------------------------------------------

print("\nHANDS OFF")

student["left_wrist"] = (20, 20)
student["right_wrist"] = (30, 30)

for _ in range(5):

    results = engine.update(student, students)

    events = alerts.update(student["student_id"], results)

    if events:
        print(events)

    time.sleep(1)


# -------------------------------------------------------------
# Reset
# -------------------------------------------------------------

student["left_wrist"] = (150, 180)
student["right_wrist"] = (200, 180)

results = engine.update(student, students)

alerts.update(student["student_id"], results)


# -------------------------------------------------------------
# Peeping
# -------------------------------------------------------------

print("\nPEEPING")

student["yaw"] = 45

for _ in range(5):

    results = engine.update(student, students)

    events = alerts.update(student["student_id"], results)

    if events:
        print(events)

    time.sleep(1)


student["yaw"] = 0

results = engine.update(student, students)

alerts.update(student["student_id"], results)


# -------------------------------------------------------------
# History
# -------------------------------------------------------------

print("\nEVENT HISTORY")

for event in alerts.get_history():
    print(event)