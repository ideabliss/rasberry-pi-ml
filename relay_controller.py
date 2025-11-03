import RPi.GPIO as GPIO
import time

# Use BCM numbering
RELAY_PIN = 17     # Controls electric relay
BUZZER_PIN = 27    # Controls buzzer
LED_PIN = 22       # Controls LED

GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_PIN, GPIO.OUT)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
GPIO.setup(LED_PIN, GPIO.OUT)

# Initialize all OFF (active LOW for relay)
GPIO.output(RELAY_PIN, 1)
GPIO.output(BUZZER_PIN, 0)
GPIO.output(LED_PIN, 0)

def trigger_alert(duration=5):
    """Trigger relay, buzzer, and LED for given duration"""
    try:
        print("🚨 Wild animal detected! Triggering alert system...")

        # Turn ON all alerts
        GPIO.output(RELAY_PIN, 0)  # Relay ON
        GPIO.output(BUZZER_PIN, 1) # Buzzer ON
        GPIO.output(LED_PIN, 1)    # LED ON

        time.sleep(duration)

        # Turn OFF all alerts
        GPIO.output(RELAY_PIN, 1)
        GPIO.output(BUZZER_PIN, 0)
        GPIO.output(LED_PIN, 0)

        print("✅ Alert sequence completed.\n")

    except Exception as e:
        print(f"❌ Relay Error: {e}")

def cleanup_relay():
    """Safely cleanup GPIO pins"""
    GPIO.output(RELAY_PIN, 1)
    GPIO.output(BUZZER_PIN, 0)
    GPIO.output(LED_PIN, 0)
    GPIO.cleanup()
    print("🧹 GPIO cleaned up safely.")
