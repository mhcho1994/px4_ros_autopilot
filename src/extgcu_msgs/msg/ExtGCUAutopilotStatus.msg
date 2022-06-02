# Autopilot status description
# This topic contains the status information from the autopilot module
# Considering UDP environment, waypoint request signal is re-initialized 
# when the autopilot module receives update signal
# External autopilot is engaged when readiness from autopilot/attitude commander module is set to 1

uint64 timestamp		            # [us] time since system start

uint8 wpt_update_request            # [-] waypoint update request
uint8 external_autopilot_ready      # [-] readiness of external autopilot