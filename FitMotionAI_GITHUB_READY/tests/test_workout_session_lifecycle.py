# -*- coding: utf-8 -*-
import unittest
import os
from pathlib import Path
from app import app, db, WorkoutExercise, User, get_or_create_shared_state, EMERGENCY_STATES

ROOT = Path(__file__).resolve().parent.parent


class WorkoutSessionLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.app = app
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def test_stop_workout_session_cleans_runtime_and_stops_background(self):
        """Test stopping workout session releases state completely so it never runs in background."""
        with self.client.session_transaction() as sess:
            sess["user_id"] = 1
            sess["user"] = {"id": 1, "username": "user", "role": "user", "name": "Nguoi dung"}

        # Simulate an active session in shared_state
        shared = get_or_create_shared_state(1, "squat")
        shared["total_rep"] = 5
        shared["active"] = True

        res = self.client.post("/nguoi-dung/api/dung-phien-tap/squat")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("success"))

        # Verify state is cleared and marked done
        self.assertTrue(shared.get("workout_done"))
        self.assertFalse(shared.get("active"))
        self.assertEqual(shared.get("total_rep", 0), 0)

    def test_finish_workout_api_with_confirm_exit_handles_zero_reps(self):
        """Test finishing workout with confirm_exit flag when zero reps recorded."""
        with self.client.session_transaction() as sess:
            sess["user_id"] = 1
            sess["user"] = {"id": 1, "username": "user", "role": "user", "name": "Nguoi dung"}

        shared = get_or_create_shared_state(1, "squat")
        shared["total_rep"] = 0

        # Normal finish without confirm_exit returns 400
        res = self.client.post("/nguoi-dung/api/ket-thuc-buoi-tap/squat", json={})
        self.assertEqual(res.status_code, 400)
        self.assertFalse(res.get_json().get("success"))

        # Finish with confirm_exit returns 200 and success
        res2 = self.client.post("/nguoi-dung/api/ket-thuc-buoi-tap/squat", json={"confirm_exit": True})
        self.assertEqual(res2.status_code, 200)
        self.assertTrue(res2.get_json().get("success"))
        self.assertFalse(res2.get_json().get("is_completed"))

    def test_all_criteria_audio_files_exist_in_static_audio(self):
        """Verify all 10 criteria WAV audio files exist in static/audio directory."""
        audio_dir = ROOT / "static" / "audio"
        self.assertTrue(audio_dir.exists())

        required_audios = [
            "criterion-1-notlow.wav",
            "criterion-2-backlean.wav",
            "criterion-3-notlow.wav",
            "criterion-4-bodyline.wav",
            "criterion-5-notbend.wav",
            "criterion-6-notstraight.wav",
            "criterion-7-elbowshift.wav",
            "criterion-8-notbend.wav",
            "criterion-9-notstraight.wav",
            "criterion-10-elbowshift.wav",
            "frame_reminder.wav",
            "wrong_side_curl.wav",
        ]
        for audio_file in required_audios:
            self.assertTrue(
                (audio_dir / audio_file).exists(),
                f"Missing audio file: {audio_file}"
            )
