

def turn_relay_on(arduino):
    arduino.write(b"1")


def turn_relay_off(arduino):
    arduino.write(b"0")
