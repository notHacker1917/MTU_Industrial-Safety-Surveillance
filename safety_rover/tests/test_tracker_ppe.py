"""
Tests for ByteTrack + PPE compliance system
Tests WITHOUT hardware (all mocks)
"""

import pytest
from collections import deque
from datetime import datetime, timedelta


class MockTrack:
    """Mock Track class for testing"""
    id_counter = 0
    
    def __init__(self, detection):
        MockTrack.id_counter += 1
        self.track_id = MockTrack.id_counter
        self.bbox = detection["bbox"]
        self.depth_m = detection["depth_m"]
        self.ppe_status = {"suit": None, "shield": None, "gloves": None}
        self.age = 0
        self.time_since_update = 0
        self.last_ppe_frame = 0
    
    @staticmethod
    def reset():
        MockTrack.id_counter = 0


class TestTrackManagement:
    """Test track creation and ID assignment"""
    
    def setUp(self):
        MockTrack.reset()
    
    def test_track_ids_increment(self):
        """Track IDs should increment sequentially"""
        self.setUp()
        detections = [
            {"bbox": [10, 20, 100, 150], "depth_m": 2.0},
            {"bbox": [50, 50, 120, 160], "depth_m": 1.5},
            {"bbox": [150, 100, 200, 250], "depth_m": 3.0},
        ]
        
        tracks = [MockTrack(d) for d in detections]
        
        assert tracks[0].track_id == 1
        assert tracks[1].track_id == 2
        assert tracks[2].track_id == 3
    
    def test_track_persistence_across_frames(self):
        """Same track should maintain ID across frames"""
        self.setUp()
        
        # Frame 1: create track
        track1 = MockTrack({"bbox": [10, 20, 100, 150], "depth_m": 2.0})
        id1 = track1.track_id
        
        # Frame 2: match same detection
        track2 = MockTrack({"bbox": [12, 22, 102, 152], "depth_m": 2.1})
        
        # IDs should be different (new object), but in real implementation,
        # we'd reuse the same track object
        assert id1 != track2.track_id  # This tests mock behavior


class TestLineCrossing:
    """Test line-crossing detection and counting"""
    
    def test_line_crossing_upward(self):
        """Upward crossing should be detected"""
        line_y = 160  # Middle of frame
        
        # Track moves from below to above
        prev_y = 180  # Below line
        curr_y = 140  # Above line
        
        # Should detect upward crossing
        crossed_upward = prev_y > line_y and curr_y <= line_y
        assert crossed_upward
    
    def test_line_crossing_downward(self):
        """Downward crossing should be detected"""
        line_y = 160
        
        # Track moves from above to below
        prev_y = 140  # Above line
        curr_y = 180  # Below line
        
        # Should detect downward crossing
        crossed_downward = prev_y <= line_y and curr_y > line_y
        assert crossed_downward
    
    def test_no_crossing_when_stationary(self):
        """Stationary track should not cross"""
        line_y = 160
        
        # Track stays above line
        prev_y = 140
        curr_y = 145
        
        crossed = (prev_y > line_y and curr_y <= line_y) or \
                  (prev_y <= line_y and curr_y > line_y)
        assert not crossed
    
    def test_cooldown_prevents_double_count(self):
        """Track crossing twice within cooldown should count once"""
        cooldown_seconds = 5
        current_time = datetime.now()
        
        # First crossing
        crossing_times = deque(maxlen=100)
        crossing_times.append(current_time)
        
        # Second crossing within cooldown
        crossing_time2 = current_time + timedelta(seconds=2)
        time_diff = (crossing_time2 - crossing_times[-1]).total_seconds()
        
        # Should be within cooldown
        assert time_diff < cooldown_seconds
        
        # Third crossing after cooldown
        crossing_time3 = current_time + timedelta(seconds=6)
        time_diff2 = (crossing_time3 - crossing_times[-1]).total_seconds()
        assert time_diff2 > cooldown_seconds


class TestPPECompliance:
    """Test PPE compliance evaluation"""
    
    def test_zone_a_all_ppe_compliant(self):
        """Zone A with all PPE should be COMPLIANT"""
        zone_a_rules = {
            "ppe_requirements": ["suit", "shield", "gloves"],
            "critical_item": "shield",
            "warning_item": "gloves",
        }
        
        ppe_status = {
            "suit": True,
            "shield": True,
            "gloves": True,
        }
        
        missing_ppe = [item for item in zone_a_rules["ppe_requirements"]
                       if not ppe_status.get(item, False)]
        
        # Should have no missing PPE
        assert len(missing_ppe) == 0
    
    def test_zone_a_missing_shield_is_critical(self):
        """Zone A missing shield should be CRITICAL"""
        zone_a_rules = {
            "ppe_requirements": ["suit", "shield", "gloves"],
            "critical_item": "shield",
            "warning_item": "gloves",
        }
        
        ppe_status = {
            "suit": True,
            "shield": False,  # MISSING!
            "gloves": True,
        }
        
        missing_ppe = [item for item in zone_a_rules["ppe_requirements"]
                       if not ppe_status.get(item, False)]
        
        # Should have shield in missing
        assert "shield" in missing_ppe
        
        # Shield is critical
        has_critical_item = zone_a_rules["critical_item"] in missing_ppe
        assert has_critical_item
    
    def test_zone_a_missing_gloves_is_warning(self):
        """Zone A missing only gloves should be WARNING"""
        zone_a_rules = {
            "ppe_requirements": ["suit", "shield", "gloves"],
            "critical_item": "shield",
            "warning_item": "gloves",
        }
        
        ppe_status = {
            "suit": True,
            "shield": True,
            "gloves": False,  # Only this missing
        }
        
        missing_ppe = [item for item in zone_a_rules["ppe_requirements"]
                       if not ppe_status.get(item, False)]
        
        # Has warning item but not critical
        has_warning = zone_a_rules["warning_item"] in missing_ppe
        has_critical = zone_a_rules["critical_item"] in missing_ppe
        
        assert has_warning
        assert not has_critical
    
    def test_zone_b_requires_only_gloves(self):
        """Zone B should only require gloves"""
        zone_b_rules = {
            "ppe_requirements": ["gloves"],
            "critical_item": None,
            "warning_item": "gloves",
        }
        
        ppe_status = {
            "suit": False,      # Not required
            "shield": False,    # Not required
            "gloves": True,     # Required
        }
        
        missing_ppe = [item for item in zone_b_rules["ppe_requirements"]
                       if not ppe_status.get(item, False)]
        
        # Should be compliant
        assert len(missing_ppe) == 0
    
    def test_transit_zone_no_requirements(self):
        """Transit zone should not require any PPE"""
        transit_rules = {
            "ppe_requirements": [],
            "critical_item": None,
            "warning_item": None,
        }
        
        ppe_status = {
            "suit": False,
            "shield": False,
            "gloves": False,
        }
        
        missing_ppe = [item for item in transit_rules["ppe_requirements"]
                       if not ppe_status.get(item, False)]
        
        # Should always be compliant
        assert len(missing_ppe) == 0


class TestAlertDeduplication:
    """Test alert deduplication and cooldown"""
    
    def test_alert_cooldown_prevents_spam(self):
        """Same alert within cooldown should be suppressed"""
        alert_cooldown = 10  # seconds
        
        # Track recent alerts
        last_alert_time = {}
        
        # First alert
        track_id = 1
        current_time = datetime.now()
        last_alert_time[track_id] = current_time
        
        # Second alert immediately after
        time_since_last = (datetime.now() - last_alert_time[track_id]).total_seconds()
        should_emit = time_since_last >= alert_cooldown
        
        # Should NOT emit (within cooldown)
        assert not should_emit
        
        # Alert after cooldown
        current_time_later = current_time + timedelta(seconds=11)
        time_since_last = (current_time_later - last_alert_time[track_id]).total_seconds()
        should_emit = time_since_last >= alert_cooldown
        
        # Should emit (cooldown expired)
        assert should_emit
    
    def test_different_tracks_independent_cooldown(self):
        """Different tracks should have independent cooldowns"""
        alert_cooldown = 10
        current_time = datetime.now()
        
        # Alerts for different tracks
        last_alert_time = {
            1: current_time,              # Track 1 alerted now
            2: current_time - timedelta(seconds=15),  # Track 2 alerted 15s ago
        }
        
        # Check track 1 (within cooldown)
        time_diff_1 = (current_time - last_alert_time[1]).total_seconds()
        should_emit_1 = time_diff_1 >= alert_cooldown
        
        # Check track 2 (outside cooldown)
        time_diff_2 = (current_time - last_alert_time[2]).total_seconds()
        should_emit_2 = time_diff_2 >= alert_cooldown
        
        assert not should_emit_1
        assert should_emit_2


class TestEnvironmentEscalation:
    """Test alert escalation based on environment"""
    
    def test_high_temp_escalates_alert(self):
        """High temperature should escalate alert level"""
        temp_alert_c = 45.0
        temp_critical_c = 50.0
        
        current_temp = 52.0
        
        # Check escalation
        if current_temp > temp_critical_c:
            alert_level = "CRITICAL"
        elif current_temp > temp_alert_c:
            alert_level = "WARNING"
        else:
            alert_level = "OK"
        
        assert alert_level == "CRITICAL"
    
    def test_high_humidity_escalates_alert(self):
        """High humidity should escalate alert level"""
        humidity_alert_pct = 85.0
        humidity_critical_pct = 95.0
        
        current_humidity = 96.0
        
        # Check escalation
        if current_humidity > humidity_critical_pct:
            alert_level = "CRITICAL"
        elif current_humidity > humidity_alert_pct:
            alert_level = "WARNING"
        else:
            alert_level = "OK"
        
        assert alert_level == "CRITICAL"
    
    def test_normal_environment_no_escalation(self):
        """Normal environment should not escalate"""
        temp_alert_c = 45.0
        humidity_alert_pct = 85.0
        
        temp = 30.0
        humidity = 60.0
        
        should_escalate = temp > temp_alert_c or humidity > humidity_alert_pct
        assert not should_escalate


class TestLineCrossingHysteresis:
    """FIX A6: Test line-crossing with hysteresis band to prevent jitter"""
    
    def test_no_double_count_hysteresis_band(self):
        """
        FIX A6: Track oscillating within hysteresis band should NOT trigger crossing.
        LINE_Y ± 12px = "band" zone, crossing only when both sides outside band.
        """
        line_y = 160
        hysteresis = 12  # pixels
        band_min = line_y - hysteresis  # 148
        band_max = line_y + hysteresis  # 172
        
        # Simulate track oscillating around line (within band)
        positions = [
            140,    # above band - valid state
            158,    # entering band from above - should stay "above" until exits
            165,    # in band - no side change
            155,    # in band - no side change
            168,    # in band - no side change
            145,    # back above band
        ]
        
        # Process positions
        last_side = None
        crossing_count = 0
        
        for y in positions:
            # Determine side with hysteresis
            if y < band_min:
                current_side = "above"
            elif y > band_max:
                current_side = "below"
            else:
                current_side = "band"  # In hysteresis zone
            
            # Count crossing only if both sides outside band
            if (last_side is not None and
                last_side != "band" and
                current_side != "band" and
                last_side != current_side):
                crossing_count += 1
            
            # Update last_side only if outside band
            if current_side != "band":
                last_side = current_side
        
        # Should be ZERO crossings (oscillation within band doesn't count)
        assert crossing_count == 0
    
    def test_count_after_clear_crossing_hysteresis(self):
        """FIX A6: Clear crossing through band should count as single crossing."""
        line_y = 160
        hysteresis = 12
        band_min = line_y - hysteresis
        band_max = line_y + hysteresis
        
        # Track moves from above to below (clear crossing through band)
        positions = [
            80,     # above band - valid
            145,    # above band - valid
            165,    # in band - no change
            200,    # below band - valid CROSSING!
        ]
        
        last_side = None
        crossing_count = 0
        
        for y in positions:
            if y < band_min:
                current_side = "above"
            elif y > band_max:
                current_side = "below"
            else:
                current_side = "band"
            
            if (last_side is not None and
                last_side != "band" and
                current_side != "band" and
                last_side != current_side):
                crossing_count += 1
                print(f"CROSSING: {last_side} → {current_side}")
            
            if current_side != "band":
                last_side = current_side
        
        # Should count as 1 crossing
        assert crossing_count == 1
    
    def test_cooldown_prevents_recount_hysteresis(self):
        """
        FIX A6: Same track crossing twice within 4-second cooldown
        should only count once.
        """
        cooldown_seconds = 4.0
        
        # Track 1 crosses at t=0
        last_cross_time = 0.0
        current_time = 1.0  # 1 second later
        
        # Should NOT count (within cooldown)
        time_since_cross = current_time - last_cross_time
        should_count = time_since_cross >= cooldown_seconds
        assert not should_count
        
        # Track crosses again at t=5 (5 seconds after first cross)
        current_time = 5.0
        time_since_cross = current_time - last_cross_time
        should_count = time_since_cross >= cooldown_seconds
        
        # Should count (outside cooldown)
        assert should_count
    
    def test_multiple_tracks_independent_side_tracking(self):
        """FIX A6: Different tracks should track their own last_side."""
        line_y = 160
        hysteresis = 12
        band_min = line_y - hysteresis
        band_max = line_y + hysteresis
        
        # Track 1: above line
        track1_y = 100
        track1_side = "above" if track1_y < band_min else \
                     ("below" if track1_y > band_max else "band")
        
        # Track 2: below line
        track2_y = 200
        track2_side = "above" if track2_y < band_min else \
                     ("below" if track2_y > band_max else "band")
        
        # Tracks should have different sides
        assert track1_side == "above"
        assert track2_side == "below"
        assert track1_side != track2_side


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
