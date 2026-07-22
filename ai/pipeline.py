"""
ai/pipeline.py

Phase 9 - Processing Pipeline

Responsibilities
----------------
- Coordinate all AI modules
- Process one CCTV frame
- Evaluate rules
- Generate alerts
- Return pipeline results
"""
from __future__ import annotations
from recorder.recorder import EvidenceRecorder
from database.crud import DatabaseCRUD
from ai.alerts import AlertLevel
from dataclasses import dataclass
import time
from typing import Any
from ai.alerts import AlertEvent
from ai.alerts import AlertManager
from ai.rules import RuleEngine
from ai.rules import StudentRuleResults

@dataclass(slots=True)
class PipelineResult:
    """
    Output produced after processing one frame.
    """

    frame: Any

    students: list[dict]

    rule_results: dict[int, StudentRuleResults]

    alerts: list[AlertEvent]

    processing_time: float

class Pipeline:
    """
    Main AI processing pipeline.

    Coordinates every module required for
    one CCTV frame.
    """
    def __init__(
        self,
        tracker,
        desk_mapper,
        orientation_estimator,
        head_pose_estimator,
        rule_engine: RuleEngine,
        alert_manager: AlertManager,
        evidence_recorder: EvidenceRecorder,
        database: DatabaseCRUD,
    ) -> None:
        """
        Initialize processing pipeline.
        """


        self._tracker = tracker

        self._desk_mapper = desk_mapper

        self._orientation_estimator = orientation_estimator

        self._head_pose_estimator = head_pose_estimator

        self._rule_engine = rule_engine

        self._alert_manager = alert_manager

        self._database = database      

        self._evidence_recorder = evidence_recorder



    def process(
        self,
        frame: Any,
    ) -> PipelineResult:
        """
        Process one CCTV frame.

        Returns
        -------
        PipelineResult
            Processed frame output.
        """
        start_time = time.perf_counter()

        timestamp = time.time()

        evidence = self._evidence_recorder.update(
            frame=frame,
            timestamp=timestamp,
        )

        if evidence is not None:

            self._database.create_evidence(
                student_id=evidence.student_id,
                event_type=evidence.event_type,
                clip_path=evidence.clip_path,
                created_at=evidence.created_at,
            )

            print(
                f"[Recorder] Evidence saved: "
                f"{evidence.clip_path}"
            )

        # ---------------------------------------------------------
        # YOLO Pose + ByteTrack Tracking
        # ---------------------------------------------------------

        tracks = self._tracker.track(frame)

        # ---------------------------------------------------------
        # Desk Assignment
        # ---------------------------------------------------------

        desk_mapping = self._desk_mapper.update(tracks)

        students = []

        for track in tracks:

            student = track.copy()

            track_id = student["track_id"]

            student["student_id"] = track_id

            student["desk_id"] = desk_mapping.get(track_id)

            keypoints = student.get("keypoints", {})
            student.update(keypoints) 

            students.append(student)

        # ---------------------------------------------------------
        # Head Orientation Estimation
        # ---------------------------------------------------------

        students = self._orientation_estimator.estimate_all(
            students,
        ) 

        # ---------------------------------------------------------
# Head Orientation Estimation
# ---------------------------------------------------------

        students = self._head_pose_estimator.estimate_all(
            frame,
            students,
        )

        

        # ---------------------------------------------------------
        # Rule Engine
        # ---------------------------------------------------------

        rule_results: dict[int, StudentRuleResults] = {}

        alerts: list[AlertEvent] = []

        # ---------------------------------------------------------
        # Evaluate Every Student
        # ---------------------------------------------------------

        for student in students:

            student_id = student.get("student_id")

            if student_id is None:
                continue

            try:

                result = self._rule_engine.update(
                    student=student,
                    students=students,
                )

                rule_results[student_id] = result

                events = self._alert_manager.update(
                    student_id=student_id,
                    results=result,
                )

                alerts.extend(events)

                for event in events:

                    self._database.create_alert(
                        student_id=student_id,
                        rule_name=event.alert_type.value,
                        alert_level=event.level.name,
                        created_at=event.timestamp,
                    )

                    if event.level == AlertLevel.RED:

                        started = self._evidence_recorder.start_recording(
                            student_id=student_id,
                            event_type=event.alert_type.value,
                        )

                        if started:
                            print(
                                f"[Recorder] Recording started "
                                f"for Student {student_id}"
                            )

                

            except Exception as exc:

                print(
                    f"[Pipeline] Student {student_id} failed: {exc}"
                )

        # ---------------------------------------------------------
        # Visualization
        # ---------------------------------------------------------

        frame = self._tracker.visualize(
            frame,
            students,
        )

        frame = self._orientation_estimator.visualize(
            frame,
            students,
        )

        processing_time = time.perf_counter() - start_time

        

        return PipelineResult(
            frame=frame,
            students=students,
            rule_results=rule_results,
            alerts=alerts,
            processing_time=processing_time,
        )

    