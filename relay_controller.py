import RPi.GPIO as GPIO
import time

# --- GPIO Configuration ---
# Using BCM numbering
RELAY_MAIN = 17    # Relay 1 - Electric Device
RELAY_BUZZER = 27  # Relay 2 - Buzzer
RELAY_LED = 22     # Relay 3 - LED

# --- GPIO Setup ---
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Setup all relay pins as outputs
for pin in [RELAY_MAIN, RELAY_BUZZER, RELAY_LED]:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.HIGH)  # Initialize OFF (active LOW relays)

def trigger_alert(duration=5):
    """
    Trigger all 3 relays (Electric Device, Buzzer, LED)
    for given duration.
    """
    try:
        print("🚨 Wild animal detected! Activating alert system...")

        # Turn ON all relays (active LOW)
        GPIO.output(RELAY_MAIN, GPIO.LOW)
        GPIO.output(RELAY_BUZZER, GPIO.LOW)
        GPIO.output(RELAY_LED, GPIO.LOW)

        time.sleep(duration)

        # Turn OFF all relays
        GPIO.output(RELAY_MAIN, GPIO.HIGH)
        GPIO.output(RELAY_BUZZER, GPIO.HIGH)
        GPIO.output(RELAY_LED, GPIO.HIGH)

        print("✅ All alerts deactivated successfully.\n")

    except Exception as e:
        print(f"❌ Error triggering relays: {e}")
        cleanup_relay()

def cleanup_relay():
    """Safely turn OFF all relays and clean up GPIO"""
    for pin in [RELAY_MAIN, RELAY_BUZZER, RELAY_LED]:
        GPIO.output(pin, GPIO.HIGH)
    GPIO.cleanup()
    print("🧹 GPIO cleaned up safely.")

# --- Test Run ---
if __name__ == "__main__":
    try:
        trigger_alert(duration=5)
    finally:
        cleanup_relay()
